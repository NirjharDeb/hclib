#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"

#include <regex>

using namespace clang;
using namespace clang::tooling;
static llvm::cl::OptionCategory MyToolCategory("my-tool options");

using namespace clang::ast_matchers;

#define MAX_DEPTH 2

namespace __internal__ {
    const CXXRecordDecl* searchLambda(const Stmt *s) {
        if (!s) { return nullptr; }
        const LambdaExpr *lambda = dyn_cast<LambdaExpr>(s);
        if (lambda) {
            return lambda->getLambdaClass();
        }
        for (auto const &p : s->children()) {
            auto q =  searchLambda(p);
            if (q) return q;
        }
        return nullptr;
    }

    void findName(const Expr *e, llvm::SmallString<16> &ret) {
        const DeclRefExpr *dre = dyn_cast<DeclRefExpr>(e);
        const IntegerLiteral *intL = dyn_cast<IntegerLiteral>(e);
        if (intL) {
            intL->getValue().toStringUnsigned(ret);
        } else {
            const ImplicitCastExpr *ice = dyn_cast<ImplicitCastExpr>(e);
            if (!dre && ice) {
                dre = dyn_cast<DeclRefExpr>(ice->getSubExpr());
                const ImplicitCastExpr *ice2 = dyn_cast<ImplicitCastExpr>(ice->getSubExpr());
                if (ice2) {
                    dre = dyn_cast<DeclRefExpr>(ice2->getSubExpr());
                }
            }
            assert(dre != str::nullptr);
            ret = dre->getDecl()->getName();
        }
        llvm::errs() << "Could not get the name of:\n";
        e->dump();
    }

    bool isArgToSend(const VarDecl *d, const Stmt *s) {
        bool ret = false;
        if (!d || !s) {
            return ret;
        }
        const CXXMemberCallExpr *send = dyn_cast<CXXMemberCallExpr>(s);
        if (send) {
            if (send->getMethodDecl()
                && send->getMethodDecl()->getName().compare("send") == 0) {
                const DeclRefExpr *dre = dyn_cast<DeclRefExpr>(send->getArg(1));
                const ImplicitCastExpr *ice = dyn_cast<ImplicitCastExpr>(send->getArg(1));
                if (!dre && ice) {
                    dre = dyn_cast<DeclRefExpr>(ice->getSubExpr());
                }
                if (dre && dre->getDecl() == d) {
                    return true;
                }
            }
        }
        for (auto const &p : s->children()) {
            ret |= isArgToSend(d, p);
        }
        return ret;
    }

    void processCaptures(const CXXRecordDecl *lambda, const unsigned int depth, std::vector<std::vector<const VarDecl *>> &arrays, std::vector<std::vector<const VarDecl *>> &scalars, std::vector<const Stmt*> &lambdas, clang::DiagnosticsEngine &DE) {
        assert(lambda->isLambda() == true && "lambda is expected");

        // Diagnostics
        const unsigned ID_LAMBDA_NONPTR = DE.getCustomDiagID(clang::DiagnosticsEngine::Remark, "[captured] non-pointer var: %0");
        const unsigned ID_LAMBDA_NONPTR_EXCLUDED = DE.getCustomDiagID(clang::DiagnosticsEngine::Remark, "[captured] non-pointer var (excluded because it's an arg to send()): %0");
        const unsigned ID_LAMBDA_PTR = DE.getCustomDiagID(clang::DiagnosticsEngine::Remark, "[captured] pointer var: %0");

        llvm::errs() << "[processCaptures] depth = " << depth << "\n";
        auto const *mDecl = lambda->getLambdaCallOperator();
        lambdas[depth] = mDecl->getBody();
        for (auto const &Capture : lambda->captures()) {
            auto const var = Capture.getCapturedVar();
            if (!var->getType().getTypePtr()->isPointerType()) {
                // if there is send API in lambda avoid args to it
                if(!__internal__::isArgToSend(var, mDecl->getBody())) {
                    DE.Report(lambda->getBeginLoc(), ID_LAMBDA_NONPTR).AddString(var->getName());
                    scalars[depth].push_back(var);
                } else {
                    DE.Report(lambda->getBeginLoc(), ID_LAMBDA_NONPTR_EXCLUDED).AddString(var->getName());
                }
            } else {
                DE.Report(lambda->getBeginLoc(), ID_LAMBDA_PTR).AddString(var->getName());
                arrays[depth].push_back(var);
            }
        }
        auto const *inner_lambda = searchLambda(mDecl->getBody());
        if (inner_lambda) {
            processCaptures(inner_lambda, depth+1, arrays, scalars, lambdas, DE);
        }

        if (depth == 0) {
            for (int i = 0; i < MAX_DEPTH; i++) {
                for (auto const &S : scalars[i]) {
                    llvm::errs() << S->getName() << "\n";
                }
            }
        }

    }

    std::pair<std::string, std::string> synthesizePacketType(const int uniqueID, const int nMBs, std::vector<std::vector<const VarDecl *>> &scalars, std::map<const VarDecl*, std::string> &nameMap) {
        llvm::errs() << scalars.size() << ", " << scalars[0].size() << "\n";
        if (nMBs == 1 && scalars[0].size() == 1) {
            // single MB & single scalar
            nameMap.insert(std::pair<const VarDecl*, std::string>(scalars[0][0], "pkt"));
            return std::make_pair(scalars[0][0]->getType().getAsString(), "primitive");
        } else {
            std::vector<const VarDecl*> done;
            std::string ret = "struct packet" + std::to_string(uniqueID) + "{\n";
            for (int i = 0; i < nMBs; i++) {
                for (auto const s: scalars[i]) {
                    if (std::find(done.begin(), done.end(), s) == done.end()) {
                        done.push_back(s);
                        nameMap.insert(std::pair<const VarDecl*, std::string>(s, llvm::Twine("pkt." + s->getName()).str()));
                        ret += s->getType().getAsString();
                        ret += llvm::Twine(" " + s->getName() + ";\n").str();
                    }
                }
            }
            ret += "};\n";
            llvm::errs() << ret;
            return std::make_pair("packet" + std::to_string(uniqueID), ret);
        }
        return std::make_pair("WRONG", "");
    }
    
    std::string translateLambdaBody(const Stmt *lambda, std::map<const VarDecl*, std::string> &nameMap, Rewriter &TheRewriter) {
        assert(lambda->isLambda() == true && "lambda is expected");
        std::string ret = Lexer::getSourceText(CharSourceRange::getTokenRange(lambda->getSourceRange()), TheRewriter.getSourceMgr(), TheRewriter.getLangOpts()).str();

        for (auto const& e: nameMap) {
            // TODO: avoid int -> packet.int
            // array subscripts 1: array[v] -> array[pkt.v]
            std::regex reg_idx("\\[\\s*" + e.first->getName().str() + "\\s*\\]");

            std::regex reg_idx2("\\[\\s*" + e.first->getName().str() + "\\s(\\+.+)\\]");

            // rhs
            std::regex reg_rhs("\\=\\s*" + e.first->getName().str() + ";");
            // lhs 1: int ret_val -> pkt.ret_val (ret_val is captured by inner-lambda)
            std::regex reg_lhs(e.first->getType().getAsString() + "\\s+" + e.first->getName().str());
            // lhs 2: v == -> pkt.v ==
            std::regex reg_lhs_eq(e.first->getName().str() + "\\s+==");
            // lhs 3: v < -> pkt.v <
            std::regex reg_lhs_lt(e.first->getName().str() + "\\s+<");
            ret = std::regex_replace(ret, reg_idx, "[" + e.second + "]");
            std::smatch sm;
            std::regex_search(ret, sm, reg_idx2);
            if (sm.size() == 2) {
                ret = std::regex_replace(ret, reg_idx2, "[" + e.second + sm.str(1) + "]");
            }
            ret = std::regex_replace(ret, reg_rhs, "= " + e.second + ";");
            ret = std::regex_replace(ret, reg_lhs, e.second);
            ret = std::regex_replace(ret, reg_lhs_eq, e.second + " ==");
            ret = std::regex_replace(ret, reg_lhs_lt, e.second + " <");
        }

        // get mailbox id
        std::smatch sm;
        std::regex_search(ret, sm, std::regex("send\\((.+),\\s*(.+),\\s*(.+)\\)"));
        std::string mailbox = sm.str(1);
        
        //
        std::regex reg_sel("\\get_selector.+;");
        ret = std::regex_replace(ret, reg_sel, "send(" + mailbox + ", pkt, sender_rank);");
        return ret;
    }

}

class SelectorLambdaHandler : public MatchFinder::MatchCallback {
public:
    SelectorLambdaHandler(Rewriter &R) : TheRewriter(R) {};

    virtual void run(const MatchFinder::MatchResult &Result) {
        ASTContext *Context = Result.Context;
        // Diagnostic
        clang::DiagnosticsEngine &DE = Context->getDiagnostics();
        const unsigned ID_LAMBDA_ELIGIBLE = DE.getCustomDiagID(clang::DiagnosticsEngine::Remark, "will be processed (outermost send+lambda)");

        // hs_ptr
        const auto ActorInstantiation = Result.Nodes.getNodeAs<DeclRefExpr>("hs_ptr");

        // hs_ptr->send();
        const auto ActorSendExpr = Result.Nodes.getNodeAs<CXXMemberCallExpr>("send");

        // selection
        if (!ActorInstantiation || !ActorSendExpr) {
            llvm::errs() << "skipped because ActorInstantiation or ActorSendExpr is NULL\n";
            return;
        }
        auto tname = ActorInstantiation->getDecl()->getType().getAsString();
        if (tname.find("hclib::Actor") == std::string::npos
            && tname.find("hclib::Selector") == std::string::npos) {
            llvm::errs() << "skipped because tname is " << tname << "(not hclib::Actor or hclib::Selector)\n";
            return;
        }

        // lambda
        auto const *Lambda = Result.Nodes.getNodeAs<CXXRecordDecl>("Lambda");
        auto const *mDecl = Lambda->getLambdaCallOperator();
        DE.Report(Lambda->getBeginLoc(), ID_LAMBDA_ELIGIBLE);

        // Analyze the captured vars by the lambda
        // 1. Non pointer type: treated as an index to global arrays
        // 2. Pointer type: treated as a global array
        // TODO: maybe some selection is required
        std::vector<std::vector<const VarDecl *>> arrays(MAX_DEPTH);
        std::vector<std::vector<const VarDecl *>> scalars(MAX_DEPTH);
        std::vector<const Stmt*> lambdas(MAX_DEPTH);

        __internal__::processCaptures(Lambda, 0, arrays, scalars, lambdas, DE);

        int depth = 0;
        for (; depth < MAX_DEPTH; depth++) {
            llvm::errs() << "[captured level-" << depth << "]\n";
            if (scalars[depth].empty()) {
                break;
            }
            for (auto const &S : scalars[depth]) {
                llvm::errs() << S->getName() << "\n";
            }
        }

        const int nMBs = depth;
        static int uniqueID = 0;
        std::map<const VarDecl*, std::string> nameMap;
        std::pair<std::string, std::string> packet;
        packet = __internal__::synthesizePacketType(uniqueID, nMBs, scalars, nameMap);
        std::string packet_type = packet.first;
        std::string packet_def = packet.second;
        llvm::errs() << "packet_type: " << packet_type << "\n";

        // Synthesize a selector class
        ActorInstantiation->getDecl()->dump();
        const VarDecl *ActorInstance = dyn_cast<VarDecl>(ActorInstantiation->getDecl());
        // Synthesize
        {
            std::string mes = "\n";
            if (packet_def.compare("primitive") != 0) {
                mes += packet_def;
            }
            const std::string CLASS = "SynthesizedActor" + std::to_string(uniqueID);
            mes += "class " + CLASS + " : public hclib::Selector<" + std::to_string(nMBs) + ", " + packet_type + "> { \n";
            mes += "public: \n";
            // output arrays
            // (outermost lambda should captures the all array)
            for (auto const v : arrays[0]) {
                mes += v->getType().getAsString() + " " + v->getName().str() + ";\n";
            }

            // lambda body
            for (unsigned int mb = 0; mb < nMBs; mb++) {
                mes += "void process" + std::to_string(mb) + "(";
                mes += packet_type + " pkt, ";
                mes += "int sender_rank)";
                mes += __internal__::translateLambdaBody(lambdas[mb], nameMap, TheRewriter);
                mes += "\n";
            }

            // Constructor
            mes += CLASS + "(";
            for (auto const v : arrays[0]) {
                mes += v->getType().getAsString() + " _" + v->getName().str();
                if (v != *(arrays[0].end() - 1)) mes += ", ";
            }
            mes += ")";
            for (auto const v : arrays[0]) {
                if (v == *(arrays[0].begin())) mes += ": ";
                mes += v->getName().str() + "(_" + v->getName().str() + ")";
                if (v != *(arrays[0].end() - 1)) mes += ", ";
                else mes += " ";
            }

            // mb[0].process = [this](pkt_type pkt, int sender_rank) { this->process(pkt, sender_rank);};
            mes += "{\n";
            for (unsigned int mb = 0; mb < nMBs; mb++) {
                mes += "mb[" + std::to_string(mb) + "].process = [this](" + packet_type + " pkt, int sender_rank) { this->process" + std::to_string(mb) + "(pkt, sender_rank); };\n";
            }
            mes += "}\n";
            mes += "};\n";

            // instantiation
            mes += CLASS + " *" + ActorInstance->getName().str();
            mes += " = new " + CLASS + "(";
            for (auto const v : arrays[0]) {
                mes += v->getName().str();
                if (v != *(arrays[0].end() - 1)) mes += ", ";
            }
            mes += ");\n";
            mes += "//";
            //ActorInstantiation->getDecl()->getBeginLoc().print(llvm::errs(), TheRewriter.getSourceMgr());
            TheRewriter.InsertText(ActorInstantiation->getDecl()->getBeginLoc(), mes, true, true);
        }
        // Update actor->send
        {
            // TODO: the arguments to send is always int + lambda?
            // Histogram: arg0 is pe
            // IG: arg1 is pe
#if 0
            const DeclRefExpr *arg0 = nullptr;
            for (auto const &p : ActorSendExpr->getArg(0)->children()) {
                for (auto const &q : p->children()) {
                    arg0 = dyn_cast<DeclRefExpr>(q);
                    if (arg0) {
                        llvm::errs() << "hoge2\n";
                        arg0->dump();
                    }
                }
            }
            if (arg0) {
                //const VarDecl argd = dyn_cast<VarDecl>(arg);
                std::string mes = ActorInstance->getName().str() + "->send(";

                // Packet creation
                // col
                for (auto const v: scalars[0]) {
                    mes += v->getName().str();
                    mes += ", ";
                }
                // pe
                mes += arg0->getDecl()->getName().str() + ");\n ";
                mes += "//";
                TheRewriter.InsertText(ActorSendExpr->getBeginLoc(), mes, true, true);
            }
#else
            const std::string PACKET = "pkt" + std::to_string(uniqueID);
            std::string mes = packet_type + " " + PACKET + ";\n";
            if (scalars[0].size() == 1) {
                mes += PACKET + " = " + llvm::Twine(scalars[0][0]->getName()).str() + ";\n";
            } else {
                for (auto const &S: scalars[0]) {
                    mes += llvm::Twine(PACKET + "." + S->getName() + " = " + S->getName() + ";\n").str();
                }
            }
            llvm::SmallString<16> orgArg0;
            llvm::SmallString<16> orgArg1;
            __internal__::findName(ActorSendExpr->getArg(0), orgArg0);
            __internal__::findName(ActorSendExpr->getArg(1), orgArg1);
            mes += ActorInstance->getName().str() + "->send(";
            mes += orgArg0.str();
            mes += ", pkt" + std::to_string(uniqueID) + ", ";
            mes += orgArg1.str();
            mes += ")";
            TheRewriter.ReplaceText(SourceRange(ActorSendExpr->getBeginLoc(), ActorSendExpr->getEndLoc()), mes);
#endif
        }
        uniqueID++;
    }

private:
    Rewriter &TheRewriter;
};

class SelectorTransConsumer : public clang::ASTConsumer {
public:
    explicit SelectorTransConsumer(Rewriter &R)
        : HandlerForLambda(R) {

        auto Lambda = expr(hasType(cxxRecordDecl(isLambda()).bind("Lambda")));
        // This matches some internal repl of ->send();
#if 1
        Matcher.addMatcher(cxxMemberCallExpr(on(declRefExpr().bind("hs_ptr")),callee(cxxMethodDecl(hasName("send"))),
                                             hasArgument(2, Lambda)).bind("send"),
                           &HandlerForLambda);
#else
        Matcher.addMatcher(cxxMemberCallExpr(callee(cxxMethodDecl(hasName("send"))),
                                             hasArgument(1, Lambda),
                                             unless(hasAncestor(cxxMemberCallExpr(callee(cxxMethodDecl(hasName("send"))))))
                               ).bind("send"),
                           &HandlerForLambda);
#endif
    }

    virtual void HandleTranslationUnit(clang::ASTContext &Context) {
        // Run the matchers when we have the whole TU parsed.
        Matcher.matchAST(Context);
    }
private:
    SelectorLambdaHandler HandlerForLambda;
    MatchFinder Matcher;
};

class SelectorTransAction : public clang::ASTFrontendAction {
public:
    void EndSourceFileAction() override {
        SourceManager &SM = TheRewriter.getSourceMgr();
        llvm::errs() << "** EndSourceFileAction for: "
                     << SM.getFileEntryForID(SM.getMainFileID())->getName() << "\n";
        TheRewriter.getEditBuffer(SM.getMainFileID()).write(llvm::outs());
    }

    virtual std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(
        clang::CompilerInstance &Compiler, llvm::StringRef InFile) {
        llvm::errs() << "** Creating AST consumer for: " << InFile << "\n";
        TheRewriter.setSourceMgr(Compiler.getSourceManager(), Compiler.getLangOpts());
        return std::unique_ptr<clang::ASTConsumer>(
            new SelectorTransConsumer(TheRewriter));
    }
private:
    Rewriter TheRewriter;
};

int main(int argc, const char **argv) {
    auto OptionsParserOrError = CommonOptionsParser::create(argc, argv, MyToolCategory);
    if (auto Err = OptionsParserOrError.takeError()) {
        llvm_unreachable("Option Error"); 
    }
    CommonOptionsParser &OptionsParser = *OptionsParserOrError;
    ClangTool Tool(OptionsParser.getCompilations(),
                   OptionsParser.getSourcePathList());

    return Tool.run(newFrontendActionFactory<SelectorTransAction>().get());
}
