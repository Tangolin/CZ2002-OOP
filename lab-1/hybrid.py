import argparse
import numpy as np
import json
import matplotlib.pyplot as plt
import time

def mergeSort(arr, l, r, cutoff):
    if l < r:
        m = l+(r-l)//2
        if len(arr[l:r+1]) > cutoff:            
            return mergeSort(arr, l, m, cutoff)+mergeSort(arr, m+1, r, cutoff)+merge(arr, l, m, r)
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
    parser.add_argument('--int', dest='interval', default=1, type=int, help='interval between iteration')
    parser.add_argument('--cutoff', dest='cutoff', default=10, type=int, help='maximum value of S')
    parser.add_argument('--type', dest='type', default='random', type=str, help='type of initial array')

    args = parser.parse_args()
    
    NUM_REPEAT = 30

    # random cases
    lengths = [i for i in range(1, args.max+1, args.interval)]
    cutoffs = [i for i in range(1, args.cutoff+1)]

    def f(length, cutoff):
        if length < cutoff:
            return np.inf

        num_comp = 0
        if args.type == 'random':
            cur_time = time.time()
            # print(cur_time)
            for _ in range(NUM_REPEAT):
                mergeSort(np.random.randint(0, 100, (length,)), 0, length-1, cutoff)
            # return num_comp / NUM_REPEAT
            return (time.time()-cur_time) / NUM_REPEAT

        elif args.type == 'sorted':
            cur_time = time.time()
            num_comp += mergeSort(np.arange(1, length+1), 0, length-1, cutoff)
            # return num_comp
            return time.time()-cur_time

        elif args.type == 'same':
            cur_time = time.time()
            num_comp += mergeSort(np.ones((length,)), 0, length-1, cutoff)
            # return num_comp
            return time.time()-cur_time

        elif args.type == 'reverse':
            cur_time = time.time()
            for _ in range(NUM_REPEAT):
                arr = np.random.randint(0, 100, (length,))
                arr[::-1].sort()
                num_comp += mergeSort(arr, 0, length-1, cutoff)
            # return num_comp / NUM_REPEAT
            return (time.time()-cur_time) / NUM_REPEAT

        else:
            return 'Invalid input'

    # time complexity analysis
    LENGTHS, CUTOFFS= np.meshgrid(lengths, cutoffs)
    KEYCOMP = []

    for length, cutoff in zip(LENGTHS.reshape(-1), CUTOFFS.reshape(-1)):
        KEYCOMP.append(f(length,cutoff))
        print(length, cutoff)

    KEYCOMP = np.array(KEYCOMP).reshape(LENGTHS.shape)

    # finding optimal value of S
    min_idx = np.argmin(KEYCOMP, axis=0)
    min_s = min_idx + 1

    plt.plot(lengths, min_s)
    state_dict = {length: min_s for length, min_s in zip(lengths, min_s.tolist())} 
    with open('state_'+str(args.max)+'_'+str(args.interval)+'_'+args.type+'.json', 'w') as f:
        json.dump(state_dict, f, indent = 4)

    plt.title('Optimal S value against length')
    plt.xlabel('length')
    plt.ylabel('optimal S value')
    plt.savefig('output_'+str(args.max)+'_'+str(args.interval)+'_'+args.type+'.jpg')
