using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;


namespace CS
{
	class ThreadNetwork
	{
		public List<Tup> dataLst;
		public int seed;
		public NNW network;
		public double minCost;

		public ThreadNetwork(string[] args, int seed)
		{
			dataLst = Program.InitData(args);
			this.seed = seed;
		}

		public void CreateTrainStop()
		{
			//Thread.CurrentThread.CurrentCulture = new System.Globalization.CultureInfo("en-US");
			
			// hyper parameters
			int batchSize = 80; // be careful that bS*nB < nData
			int nBatch = 4; // so there is data left for testing
			int maxCostRise = 25; // early stopping parameter
			double learningRate = 0.5;

			if (dataLst == null)
				return;
			List<Tup> lst = new List<Tup>(dataLst);

			// for stochastic gd, create mini batches for training
			Random rand = new Random(seed);
			List<Tup>[] batch = new List<Tup>[nBatch];
			for (int b = 0; b < nBatch; b++)
			{
				batch[b] = new List<Tup>();
				for (int i = 0; i < batchSize; i++)
				{
					int id = rand.Next(0, dataLst.Count);
					batch[b].Add(dataLst[id]);
					dataLst.RemoveAt(id);
				}
			}

			// create the network, and init early stopping variables
			network = new NNW(seed, new int[] {30, 20, 10, 1});

			minCost = double.PositiveInfinity;
			int costRise = 0;

			// train the network with early stopping
			for (int i = 0; i < 10000; i++)
			{
				network.Train(batch[i % nBatch], learningRate);
				//double cost = network.Cost(lst);
				double cost = network.Cost(dataLst);
				if (cost < minCost)
				{
					costRise = 0;
					minCost = cost;
					network.PackModel();
				}
				else
					costRise++;
				if (costRise >= maxCostRise)
					break ;

				if (i % 500 == 0)
					System.Console.WriteLine($"{Thread.CurrentThread.Name}, cost: " +  cost);
			}
			network.UnpackModel();
			System.Console.WriteLine($"{Thread.CurrentThread.Name}, Final cost: {minCost}");
		}

	}

		// #####################################################################
		// ####							Program								####
		// #####################################################################

	class Program
	{
		static private double[] ParseData(string[] sp)
		{
			double[] data = new double[30];

			for (int i = 0; i < 30; i++)
				data[i] = sp[i].Contains(".") ? double.Parse(sp[i]) : int.Parse(sp[i]); // need to s a n i t i z e
			return data;
		}

		static public List<Tup> InitData(string[] args)
		{
			List<Tup> lst = new List<Tup>();

			if (args.Length < 1)
				{ Console.WriteLine("Provide data"); return null; }
			try
			{
				using (StreamReader sr = new StreamReader(args[0]))
				{
					string s = sr.ReadLine();
					int	id = 0;

					while (s != null)
					{
						string[] sp = s.Split(',');
						lst.Add(new Tup(id++, sp[1] == "M", ParseData(sp.Skip(2).ToArray())));
						s = sr.ReadLine();
					}
				}
			}
			catch (IOException)
				{ Console.WriteLine("Couldn't open file >{0}<", args[0]); Environment.Exit(0); }

			// standardize it between [0, 1]
			for (int j = 0; j < 30; j++)
			{
				double min = lst.Min(x => x.data[j]);
				double max = lst.Max(x => x.data[j]) - min;
				for (int i = 0; i < lst.Count; i++)
				{
					lst[i].data[j] -= min;
					lst[i].data[j] /= max;
				}
			}
			return (lst);
		}

// #############################################################################
// #############################################################################
// #############################################################################


		static void MainTrain(string[] args)
		{
			Random rand = new Random(69); // seed 69 on 15 models
			int	nModels = 4;

			ThreadNetwork[] tn = new ThreadNetwork[nModels];
			Thread[] tr = new Thread[nModels];

			for (int i = 0; i < nModels; i++)
			{
				tn[i] = new ThreadNetwork(args, rand.Next());
				tr[i] = new Thread(tn[i].CreateTrainStop);
				tr[i].Name = "Thread " + i;
			}
			for (int i = 0; i < nModels; i++)
				tr[i].Start();
			for (int i = 0; i < nModels; i++)
				tr[i].Join();


			double minc = tn.Min(x => x.minCost);
			ThreadNetwork tnFinal = tn.First(x => x.minCost == minc);

			System.Console.WriteLine($"> General min cost: {tnFinal.minCost}");
			tnFinal.network.model.SaveModel("model");
		}

		static void MainPredict(string[] args)
		{
			Model model = new Model("model");
			NNW network = new NNW(model);
			List<Tup> lst = InitData(args);

			System.Console.WriteLine($"> cost: {network.Cost(lst)}");
		}

		static void Main(string[] args)
		{
			Thread.CurrentThread.CurrentCulture = new System.Globalization.CultureInfo("en-US");

			if (args.Length != 2 || (args.Length == 2 && args[1] != "train" && args[1] != "predict"))
			{
				System.Console.WriteLine("Usage: dotnet run [data.csv] [train/predict]");
				return ;
			}
			if (args[1] == "train")
				MainTrain(args);
			else if (args[1] == "predict")
				MainPredict(args);
		}
	}
}
