
def main():
	import os
	from pyEPOCH import inputDeck,inputBlock,run,readDeckFile
	
	a = inputDeck(dims=1)

	control = inputBlock({
		'nx'     :100,
		'nsteps' :10,
		'x_min'  :0.0,
		'x_max'  :1e-6
	},blockName='control')

	boundaries = inputBlock({
		'bc_x_min':'periodic',
		'bc_x_max':'periodic'
	},blockName='boundaries')

	fields = inputBlock({
		'ex': 3,
	},blockName='fields')

	a.addBlock(control)
	a.addBlock(boundaries)
	a.isValid()
	print(a)
	#a.removeBlocks('control',1)
	#print type(inputDeck)

	#a.writeToFile('./input.deck')

	run(a,'./data',8)

	b = readDeckFile('./data/input.deck')
	print(b)

	print(a == b)
	b += fields
	print(a == b)
	print(a.__dict__)
	print(b.__dict__)

if __name__ == "__main__":
	main()
