import math

class DataClass :
	'v1.1 of DataClass, a class to store, format and analyze data.'
	'header: tuple of strings containing the name of all the features'
	'col : number of features'
	'row : number of elements'
	'data : list of tuples, each tuple being an element'
	'sorted : list of lists, each list being sorted numeric features'
	'meta : list of dicts, describing the numeric features'
	
	# constructor, call 2 subconstructors for data or file parsing
	def __init__(self, filename = 0, data = 0, header = 0) :
		self.header = ()
		self.col = 0
		self.row = 0
		self.data = []
		self.sorted = []
		self.meta = []

		if (filename != 0 and type(filename) == str) :
			self.__init_file(filename)
		elif (data != 0 and header != 0) :
			self.__init_data(data, header)
		else :
			print("Created empty object")

	# subconstructor file, take a filename and parse all the data
	def __init_file(self, filename = "") :
		file = 0
		sp = []
		line = ""

		try :
			file = open(filename, "r")
		except IOError :
			print("Wrong file duh.")
			return
		sp = file.readline().split(',')
		self.__parse_header(sp)
		line = file.readline()
		# read all lines
		while (line) :
			sp = line.split(',')
			self.data.append(self.__parse_tuple(sp))
			line = file.readline()
		self.row = len(self.data)
		# parse the data in sorted
		for i in range(self.col) :
			self.sorted.append([])
			for j in range(self.row) :
				if (type(self.data[j][i]) == float) :
					self.sorted[i].append(self.data[j][i])
			self.sorted[i].sort()

	# subconstructor data, take a list of tuples (the data), and a tuple (the
	# header)
	def __init_data(self, data, header) :
		self.header = header
		self.data = data
		self.col = len(header)
		self.row = len(data)
		for i in range(self.col) :
			self.sorted.append([])
			for j in range(self.row) :
				if (type(self.data[j][i]) == float) :
					self.sorted[i].append(self.data[j][i])
			self.sorted[i].sort()

	# takes the header as a list, tuple it and init the instance with it
	def __parse_header(self, sp) :
		#stripping to remove newline
		for i in range(len(sp)) :
			sp[i] = str(sp[i]).strip()
		self.header = tuple(sp)
		self.col = len(self.header)


	# takes a row of data as a list of str, convert them to their type and
	# tuple it
	def __parse_tuple(self, sp) :
		for i in range(len(sp)) :
			s = str(sp[i]).strip()
			if (s.isdigit()) :
				sp[i] = int(s)
			else :
				try :
					if (s.casefold() != "nan") :
						sp[i] = float(s)
				except ValueError :
					pass
		return (tuple(sp))


	# calculate all the metadata from the sorted data
	def init_meta(self) :
		for d in self.sorted :
			meta = {
			'count' : 0, 'mean' : 0, 'std' : 0, 'min' : 0,
			'q25' : 0, 'q50' : 0, 'q75' : 0, 'max' : 0}
			mean = 0
			std = 0

			meta['count'] = len(d)
			if (meta['count'] == 0) :
				self.meta.append(meta)
				continue
			for e in d :
				mean += e
			mean /= meta['count']
			meta['mean'] = mean
			for e in d :
				std += (e - mean) ** 2
			meta['std'] = math.sqrt(std / meta['count'])
			meta['min'] = self.quartile(d, 0)
			meta['q25'] = self.quartile(d, 0.25)
			meta['q50'] = self.quartile(d, 0.50)
			meta['q75'] = self.quartile(d, 0.75)
			meta['max'] = self.quartile(d, 1)
			self.meta.append(meta)

	# quartile function
	def quartile(self, lst, quartile) :
		if (len(lst) == 0) :
			return (0.0)
		if(quartile > 1) :
			quartile = 1
		if(quartile < 0) :
			quartile = 0
		i = int(len(lst) * quartile)
		if (i >= len(lst)) :
			i = len(lst) - 1
		return (lst[i])


	# describe the data, metadata must be initialized
	def describe(self):
		if (len(self.meta) == 0) :
			print("No metadata")
			return
		self.__disp_header()
		self.__meta_line_print("Count:", "count")
		self.__meta_line_print("Mean: ", "mean")
		self.__meta_line_print("Std:  ", "std")
		self.__meta_line_print("Min:  ", "min")
		self.__meta_line_print("25%:  ", "q25")
		self.__meta_line_print("50%:  ", "q50")
		self.__meta_line_print("75%:  ", "q75")
		self.__meta_line_print("Max:  ", "max")

	def __disp_header(self) :
		print("      ", end='|')
		for i in self.header :
			print("%11.11s" % i, end = '|')
		print()

	def __meta_line_print(self, title, id) :
		print(title, end = '|')
		for i in range(self.col) :
			print("%11.3f" % self.meta[i][id], end = '|')
		print()


	# yeah, nothing much to say
	def disp(self) :
		print(f"This dataset has {self.row} attributes with {self.col} features in each\n")
		self.describe()
