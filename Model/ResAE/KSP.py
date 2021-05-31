from itertools import product
from statistics import stdev
k = [11, 5, 3, 3, 3]
s = [4, 1, 1, 1, 1]
p = [2, 2, 1, 1, 1]
mk = [3,3,1,1,3]
ms = [2,2,1,1,2]

x = 251

for i in range(5):
	x = int((x + 2*p[i] - k[i])/s[i] + 1)
	print(x)
	x = int(((x - mk[i])/ms[i] + 1))
	print(x)

in_dim = 250
S = [1,2]
K = [3,4,5,6]
P = [1,2]

Hyp = list(product(K, S, P))
Hyp = [h for h in Hyp if h[0]>h[1] and h[2]<h[0]]

encoder_params = {}
param = {}
hidens = [{(0,0):6}]
for i in range(6):
	hidens.append({})
	for key, value in hidens[i].items():
		for h in Hyp:
			hiden = value*h[1] - h[1] - 2*h[2] + h[0]
			k= list(key)
			k.append((h, hiden))
			if hiden < in_dim and value < hiden and hiden > 0:
				hidens[-1][tuple(k)] = hiden
			elif hiden == in_dim and i == 5:
				sd = []
				for item in list(k[2:]):
					sd.append(item[1])
				encoder_params[tuple(k[2:])] = sum(list(sd))
	if hidens[-1] == {}:
		break

textfile = open("params_decoder.txt", "w")
encoder_params = dict(sorted(encoder_params.items(), key=lambda item: item[1]))
for key, element in encoder_params.items():
	textfile.write('(')
	textfile.write(', '.join(tuple(map(str, key))))
	textfile.write(')')
	textfile.write(str(element))
	textfile.write("\n")
textfile.close()
