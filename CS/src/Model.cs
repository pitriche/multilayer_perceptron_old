using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using System.IO;


namespace CS
{
	/*
	** Model storing class
	*/
	public class Model
	{
		public int[] layersSize;

		public double[] weightsBiases;

		public Model(int[] layersSize)
		{
			this.layersSize = layersSize;
			this.weightsBiases = new double[CalcTotal(layersSize)];
		}

		public Model(string fileName)
		{
			try
			{
				using (StreamReader sr = new StreamReader(fileName))
				{
					string[] sp = sr.ReadLine().Trim().Split(' ');
					layersSize = new int[sp.Length];
					for (int i = 0; i < sp.Length; i++)
						layersSize[i] = int.Parse(sp[i]);

					int nbWeightsBiases = CalcTotal(layersSize);
					weightsBiases = new double[nbWeightsBiases];
					for (int i = 0; i < nbWeightsBiases; i++)
						weightsBiases[i] = double.Parse(sr.ReadLine());
				}
			}
			catch (IOException)
				{ Console.WriteLine("Couldn't open file >{0}<", fileName); Environment.Exit(0); }
		}

		public void SaveModel(string fileName)
		{
			using (StreamWriter sw = new StreamWriter(fileName))
			{
				foreach (int i in layersSize)
					sw.Write($"{i} ");
				sw.WriteLine();
				foreach (double d in weightsBiases)
					sw.WriteLine(d);
			}
		}

		public void LoadModel(string fileName)
		{
			using (StreamReader sr = new StreamReader(fileName))
			{
				string[] sp = sr.ReadLine().Trim().Split(' ');
				layersSize = new int[sp.Length];
				for (int i = 0; i < sp.Length; i++)
					layersSize[i] = int.Parse(sp[i]);

				int nbWeightsBiases = CalcTotal(layersSize);
				for (int i = 0; i < nbWeightsBiases; i++)
					weightsBiases[i] = double.Parse(sr.ReadLine());
			}
		}

		public int CalcTotal(int[] layersSize)
		{
			int nbWeightsBiases = 0;
			for (int i = 1; i < layersSize.Length; i++)
				nbWeightsBiases += layersSize[i] + layersSize[i] * layersSize[i - 1];
			return (nbWeightsBiases);
		}
	}
}
