import sys
sys.path.insert(1, 'Class')
import DataClass as dtc


if (len(sys.argv) < 2) :
	print("Give a file")
	exit()
filename = sys.argv.pop(1)

d = dtc.DataClass(filename)
d.init_meta()
d.describe()
