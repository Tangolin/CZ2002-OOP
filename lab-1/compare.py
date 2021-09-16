import argparse
import copy
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import time

NUM_REPEAT = 20

def mergeSort(arr, l, r):
    if l < r:
        m = l+(r-l)//2
 
        return mergeSort(arr, l, m)+mergeSort(arr, m+1, r)+merge(arr, l, m, r)
    
    return 0

def hybridSort(arr, l, r, cutoff):
    if l < r:
        m = l+(r-l)//2
        if len(arr[l:r+1]) > cutoff:            
            return hybridSort(arr, l, m, cutoff)+hybridSort(arr, m+1, r, cutoff)+merge(arr, l, m, r)
        else:
            return insertionSort(arr, l, r+1)
    
    return 0

def merge(arr, l, m, r):
    comp = 0
    n1 = m - l + 1
    n2 = r - m

    L = [0] * (n1)
    R = [0] * (n2)

    for i in range(0, n1):
        L[i] = arr[l + i]

    for j in range(0, n2):
        R[j] = arr[m + 1 + j]

    i = 0
    j = 0
    k = l

    while i < n1 and j < n2:
        if L[i] <= R[j]:
            arr[k] = L[i]
            i += 1
        else:
            arr[k] = R[j]
            j += 1
        k += 1
        comp +=1

    while i < n1:
        arr[k] = L[i]
        i += 1
        k += 1

    while j < n2:
        arr[k] = R[j]
        j += 1
        k += 1
    
    return comp

def insertionSort(arr, l, r):
    comp = 0
    for i in range(l+1, r):
        key = arr[i]
 
        j = i-1
        while j >= l:
            if key >= arr[j]:
                comp += 1
                break
            arr[j + 1] = arr[j]
            j -= 1
            comp += 1
        arr[j + 1] = key
    return comp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max', dest='max', default=10, type=int, help='maximum length to test')
    parser.add_argument('--cutoff', dest='cutoff', default=10, type=int, help='maximum value of S')
    parser.add_argument('--type', dest='type', default='random', choices=['random', 'sorted', 'reverse'], type=str, help='type of initial array')

    args = parser.parse_args()

    lengths = np.linspace(args.cutoff, args.max, 10, dtype=int)

    def f(length, cutoff):
        if length < cutoff:
            return np.inf

        num_comp_hybrid = 0
        num_comp_merge = 0

        if args.type == 'random':
            arr = np.random.randint(0, length, (length,))

            cur_time = time.time()
            for _ in range(NUM_REPEAT):
                num_comp_hybrid += hybridSort(copy.deepcopy(arr), 0, length-1, cutoff)
            hybrid_time = (time.time() - cur_time) / NUM_REPEAT

            cur_time = time.time()
            for _ in range(NUM_REPEAT):
                num_comp_merge += mergeSort(copy.deepcopy(arr), 0, length-1)
            merge_time = (time.time() - cur_time) / NUM_REPEAT
            
            return hybrid_time, merge_time, num_comp_hybrid / NUM_REPEAT, num_comp_merge / NUM_REPEAT

        elif args.type == 'sorted':
            arr = np.arange(1, length+1)

            cur_time = time.time()
            num_comp_hybrid += hybridSort(copy.deepcopy(arr), 0, length-1, cutoff)
            hybrid_time = (time.time() - cur_time)

            cur_time = time.time()
            num_comp_merge += mergeSort(copy.deepcopy(arr), 0, length-1)
            merge_time = (time.time() - cur_time)

            return hybrid_time, merge_time, num_comp_hybrid, num_comp_merge

        elif args.type == 'reverse':
            arr = np.random.randint(0, length, (length,))
            arr[::-1].sort()

            cur_time = time.time()
            for _ in range(NUM_REPEAT):
                num_comp_hybrid += hybridSort(copy.deepcopy(arr), 0, length-1, cutoff)
            hybrid_time = (time.time() - cur_time) / NUM_REPEAT

            cur_time = time.time()
            for _ in range(NUM_REPEAT):
                num_comp_merge += mergeSort(copy.deepcopy(arr), 0, length-1)
            merge_time = (time.time() - cur_time) / NUM_REPEAT
            
            return hybrid_time, merge_time, (num_comp_hybrid / NUM_REPEAT), (num_comp_merge / NUM_REPEAT)

        else:
            return 'Invalid input'
        
    hybrid_times = []
    merge_times = []
    num_comp_hybrids = []
    num_comp_merges = []

    for length in tqdm.tqdm(lengths):
    # for length in lengths:
        hybrid_time, merge_time, num_comp_hybrid, num_comp_merge = f(length, args.cutoff)
        hybrid_times.append(hybrid_time)
        merge_times.append(merge_time)
        num_comp_hybrids.append(num_comp_hybrid)
        num_comp_merges.append(num_comp_merge)

    plt.figure(0)
    plt.plot(lengths, hybrid_times, label='hybrid')
    plt.plot(lengths, merge_times, label='merge')
    plt.legend()
    plt.title('Performance(CPU Time)')
    plt.savefig(str(args.max)+'_CPU_Time_'+args.type+'.png')

    plt.figure(1)
    plt.plot(lengths, num_comp_hybrids, label='hybrid')
    plt.plot(lengths, num_comp_merges, label='merge')
    plt.legend()
    plt.title('Performance(Key Comparisons)')
    plt.savefig(str(args.max)+'_KeyComp_'+args.type+'.png')
