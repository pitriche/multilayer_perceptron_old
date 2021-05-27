using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;


namespace CS
{
	public struct Tup
	{
		public int id;
		public bool mal;
		public int y;
		public double[] data;

		public Tup(int id, bool mal, double[] data) : this()
		{
			this.id = id;
			this.mal = mal;
			this.y = mal ? 1 : 0;
			this.data = data;
		}
	}
}
