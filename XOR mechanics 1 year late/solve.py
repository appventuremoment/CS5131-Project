from pwn import xor
with open('output.txt', 'rb') as f:
    flag = f.read()

a = flag[0:len(flag) // 3]
b = flag[len(flag) // 3:2 * len(flag) // 3]
c = flag[2 * len(flag) // 3:]

c = xor(c, int(str(len(flag))[0]) * int(str(len(flag))[1]))
c = xor(b, c)
b = xor(a, b)
a = xor(c, a)

c = xor(b, c)
b = xor(a, b)
a = xor(a, int(str(len(flag))[0]) + int(str(len(flag))[1]))

# b1 = xor(c, int(str(len(flag))[0]) * int(str(len(flag))[1]))
# c1 = xor(b1, a)
# a1 = xor(xor(b, c1), int(str(len(flag))[0]) + int(str(len(flag))[1]))

enc = a + b + c
with open('flag.txt', 'wb') as f:
    f.write(enc)

# it is literally writing the code in the reverse order (bottom up)