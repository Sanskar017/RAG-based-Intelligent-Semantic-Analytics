import sys

# Retained your function name, but modified it to return a dictionary of {factor: exponent}
def prime_factors(num):
    factors = {}

    # Count number of 2s
    if num % 2 == 0:
        count = 0
        while num % 2 == 0:
            count += 1
            num //= 2
        factors[2] = count

    i = 3
    while i * i <= num:
        if num % i == 0:
            count = 0
            while num % i == 0:
                count += 1
                num //= i
            factors[i] = count
        i += 2

    if num > 1:
        factors[num] = 1

    return factors


# Input (Retained your exact input parsing)
n = int(sys.stdin.readline().strip())

arr = []
if n > 0:
    arr = list(map(int, sys.stdin.readline().split()))

num = int(sys.stdin.readline().strip())

# If array is empty (Note 1)
if n == 0 or len(arr) == 0:
    print(-1)
    sys.exit()

# Edge case: if num has no prime factors
if num < 2:
    print(0)
    sys.exit()

# Get prime factors map
factors = prime_factors(num)

# Calculate sum (Retained your loop and variables, adding the exponent multiplier)
ans = 0
found_any_factor_as_index = False

for idx in factors:
    if 0 <= idx < n:
        exponent = factors[idx]  # Retrieve the exponent for this prime factor
        ans += exponent * arr[idx]  # Multiply by the exponent as required by the formula
        found_any_factor_as_index = True

# Note 2: Print 0 if no factor falls within the array bounds
if found_any_factor_as_index:
    print(ans)
else:
    print(0)