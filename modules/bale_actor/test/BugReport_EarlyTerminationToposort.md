# Bug Report: Early Termination of `initiate_global_done` in `toposort_pure_selector`

## **Description**
We hypothesize that `initiate_global_done` is terminating too early, resulting in non-deterministic behavior when replacing `done(0)` with `initiate_global_done` in `toposort_pure_selector`. This issue causes early termination for certain Processor Elements (PE) and rows-per-thread combinations.

For example, with 2 PEs and 5 rows per thread, the program fails to produce the expected output, as shown below.

### **Example Output**
```shell
root@067855ca6053:~/hclib/modules/bale_actor/test# $OSHRUN -n 2 ./toposort_selector -n 5
WARNING: Failed dynamically loading /root/hclib/hclib-install/lib/libhclib_bale_actor.so for "bale_actor" dependency
WARNING: HCLIB_LOCALITY_FILE not provided, generating sane default locality information
WARNING: HCLIB_WORKERS provided, creating locale graph based on 1 workers
WARNING: Failed dynamically loading /root/hclib/hclib-install/lib/libhclib_bale_actor.so for "bale_actor" dependency
WARNING: HCLIB_LOCALITY_FILE not provided, generating sane default locality information
WARNING: HCLIB_WORKERS provided, creating locale graph based on 1 workers
Running toposort on 2 threads
buf_cnt (stack size)           (-b)   1024
Number of rows per thread      (-n)   5
Avg # of nonzeros per row      (-Z)   10.00
Erdos-Renyi edge probability   (-e)   1.000000
task mask (M) = 15 (should be 1,2,4,8,16 for agi, exstack, exstack2, conveyor, alternate
generate ER graph time 0.010902
generate perms time 0.017784
permute matrix time 0.006032
Input matrix has 10 rows and 53 nonzeros
Run toposort on mat (and tmat) ...
 Selector: 
Total toposort messages *sent*: 63
Total toposort messages *received*: 62
     0.024 seconds
ERROR: check_is_triangle fails

ERROR: After toposort_matrix_selector: mat2 is not upper-triangular!

ERROR: After toposort_matrix_selector: mat2 is not upper-triangular!
```

## **Evidence**
We have collected the following evidence to demonstrate that `initiate_global_done` is terminating too early, and should behave equivalently to `done(0)`:

### **Differences in Variable Updates:**

- Some variables, such as `lrowcnt[3]`, `lrowsum[3]`, `level[3]`, and `num_levels`, are missing the "final update" to reach their expected values.
- Diffs for values after `hclib::finish` for 2 PEs and 5 Rows per PE:
    - [PE 0 Differences](https://www.diffchecker.com/OstQqK4Y/): Differences in `lrowsum[3]`, `lrowcnt[3]`, `level[3]`, and `num_levels`. 
        - Essentially, `lrowsum[3]` and `lrowcnt[3]` are supposed to zero out, but they do not. 
        - `level[3]` is missing 1 final update to reach its correct value.
        - As a result, `num_levels` is incorrect as well.
    - [PE 1 Differences](https://www.diffchecker.com/V366KemG/): No differences found.

### **Message Count Discrepancy:**

The number of messages sent and received in the pure and modified versions was analyzed:
- Pure Toposort:
    - Messages sent: 63
    - Messages received: 63
- Modified Toposort (with `initiate_global_done`):
    - Messages sent: 63
    - Messages received: 62

The missing final message indicates that global termination occurs prematurely, supporting the hypothesis that `initiate_global_done` is not functioning as intended.

## **Steps to Reproduce**
1. Run the [toposort_selector test](https://github.com/NirjharDeb/hclib/blob/nirjhar/toposort-global-termination-v2/modules/bale_actor/test/toposort_selector.cpp) with 2 PEs and 5 rows per thread:
    ```shell
    $OSHRUN -n 2 ./toposort_selector -n 5
    ```

2. Observe the error output:
    - Non-determinism and incorrect results (e.g., matrix is not upper-triangular).
    - Message count discrepancy (sent: 63, received: 62).

3. Compare against the [toposort_pure_selector test](https://github.com/NirjharDeb/hclib/blob/nirjhar/toposort-global-termination-v2/modules/bale_actor/test/toposort_pure_selector.cpp) with 2 PEs and 5 rows per thread:
    ```shell
    $OSHRUN -n 2 ./toposort_pure_selector -n 5
    ```

4. Observe the correct output:
    - No message count discrepancy (sent: 63, received: 63).

## **Conclusion**
The `initiate_global_done` function should ensure global termination occurs after all messages are sent and received, replicating the behavior of `done(0)`.

However, the `initiate_global_done` function prematurely terminates, causing:
- Missing updates to key variables (`lrowcnt`, `lrowsum`, etc.).
- Non-deterministic results for specific PE and rows-per-thread combinations.
- Message count mismatch (1 final message not received).
