#!/usr/bin/env python

import math
import os
import shutil
import re
from itertools import groupby
from operator import itemgetter

from sdfUtils import sdfUtils

def getRanges(nums):
	'''
	Converts a sorted list of numbers into a list of consecutive ranges and numbers

	Yes, it's incomprehensible. But it works. Ripped off SO:
	https://stackoverflow.com/a/2154437
	'''
	ranges = []
	for key, group in groupby(enumerate(nums), lambda x: x[0]-x[1]):
		group = (map(itemgetter(1), group))
		group = list(map(int,group))
		if len(group) > 1:
			ranges.append((group[0], group[-1]))
		else:
			ranges.append(group[0])

	return ranges

def main():
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('dir')
	args = parser.parse_args()

	files = sdfUtils.listFiles(args.dir)

	prefixes = set()
	for f in files:
		prefixes |= set([sdfUtils.getPrefix(f)])

	prefixes = list(prefixes)
	prefixes.sort()

	counts = {}
	for p in prefixes:
		counts[p] = len(sdfUtils.listFiles(args.dir,prefix=p))

	ranges = {}
	for p in prefixes:
		nums = [ sdfUtils.getNum(f,p) for f in sdfUtils.listFiles(args.dir,p) ]
		ranges[p] = getRanges(nums)

	for p in prefixes: print('{p}: {c} files, ranges: {r}'.format(p=p,c=counts[p],r=ranges[p]))

if __name__ == '__main__':
	main()
