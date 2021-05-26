from itertools import product
from statistics import stdev
S = [1,2]
K = [2,3,4,5]
P = [0,1,2]

Hyp = list(product(K, S, P))
Hyp = [h for h in Hyp if h[0]>h[1] and h[2]<h[0]]

in_dim = 50

encoder_params = {}
param = {}
hidens = [{(0,0):in_dim}]
i = 0
for i in range(5):
	hidens.append({})
	for key, value in hidens[i].items():
		for h in Hyp:
			if h[0] <= value:
				hiden = (value + 2*h[2] - h[0])/h[1] + 1
				k= list(key)
				k.append((h, hiden))
				if hiden > 1 and value > hiden and int(hiden) == hiden:
					hidens[-1][tuple(k)] = hiden
				elif hiden == 1 and i == 4:
					sd = []
					for item in list(k[2:]):
						sd.append(item[1])
					encoder_params[tuple(k[2:])] = sum(list(sd))
	if hidens[-1] == {}:
		break

textfile = open("params_encoder.txt", "w")
encoder_params = dict(sorted(encoder_params.items(), key=lambda item: item[1]))
for key, element in encoder_params.items():
	'''
	while type(element[1]) is tuple:
		textfile.write('(')
		textfile.write(', '.join(tuple(map(str, element[0]))))
		textfile.write(')')
		element = element[1]'''
	textfile.write('(')
	textfile.write(', '.join(tuple(map(str, key))))
	textfile.write(')')
	textfile.write(str(element))
	textfile.write("\n")
textfile.close()

S = [1,2]
K = [2,3,4,5]
P = [1,2]

Hyp = list(product(K, S, P))
Hyp = [h for h in Hyp if h[0]>h[1] and h[2]<h[0]]

encoder_params = {}
param = {}
hidens = [{(0,0):1}]
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
	'''
	while type(element[1]) is tuple:
		textfile.write('(')
		textfile.write(', '.join(tuple(map(str, element[0]))))
		textfile.write(')')
		element = element[1]'''
	textfile.write('(')
	textfile.write(', '.join(tuple(map(str, key))))
	textfile.write(')')
	textfile.write(str(element))
	textfile.write("\n")
textfile.close()
			
