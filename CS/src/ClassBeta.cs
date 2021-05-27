using System;
using System.Collections;
using System.Collections.Generic;

namespace CS
{
	/*
	 * beta version of the class, broken
	 * input layer: 30 inputs
	 * 2 hidden layers, 20 neurons each
	 * output layer: 2 outputs
	 */
	public class NNWbeta
	{
		// l1w[currentNeuron, previousNeuron]
		// l1b[currentNeuron];
		private double[,] l1w; // input to layer 1 weights and biases
		private double[] l1b;

		private double[,] l2w; // layer 1 to layer 2 weights and biases
		private double[] l2b;
		
		private double[,] louw; // layer 2 to output weights and biases
		private double[] loub;


		private double[] l1; // temporary layers for use during the execution
		private double[] l2;
		private double[] lou;


		private double[,] dl1w; // input to layer 1 weights and biases derivatives
		private double[] dl1b;

		private double[,] dl2w; // layer 1 to layer 2 weights and biases derivatives
		private double[] dl2b;
		
		private double[,] dlouw; // layer 2 to output weights and biases derivatives
		private double[] dloub;


		private double[] dcz; // derivatives of the cost to the weighted sum
		private double[] dcal2; // derivatives of the cost to the l2 outputs
		private double[] dcal1; // derivatives of the cost to the l1 outputs



		private Func<double, double> Activ; // neuron activation function

		// #####################################################################
		// ####							Constructors						####
		// #####################################################################

		public NNWbeta()
		{
			l1w = new double[20, 30]; // layer 1, inputs
			l1b = new double[20];
			l2w = new double[20, 20]; // layer 2, layer 1
			l2b = new double[20];
			louw = new double[2, 20]; // outputs, layer 2
			loub = new double[2];

			l1 = new double[20];
			l2 = new double[20];
			lou = new double[2];

			dl1w = new double[20, 30]; // derivatives layer 1, inputs
			dl1b = new double[20];
			dl2w = new double[20, 20]; // derivatives layer 2, layer 1
			dl2b = new double[20];
			dlouw = new double[2, 20]; // derivatives outputs, layer 2
			dloub = new double[2];

			dcz = new double[2];	// derivatives for the chain rule
			dcal1 = new double[20];
			dcal2 = new double[20];

			Activ = x => 1 / (1 + Math.Pow(Math.E, -x)); // sigmoid activation func
		}

		public NNWbeta(int seed) : this()
		{
			Random rand = new Random(seed);

			for (int j = 0; j < 20; j++)
			{
				l1b[j] = rand.NextDouble() * 2 - 1.0;
				for (int i = 0; i < 30; i++)
					l1w[j, i] = rand.NextDouble() * 2 - 1.0;
			}
			for (int j = 0; j < 20; j++)
			{
				l2b[j] = rand.NextDouble() * 2 - 1.0;
				for (int i = 0; i < 20; i++)
					l2w[j, i] = rand.NextDouble() * 2 - 1.0;
			}
			for (int j = 0; j < 2; j++)
			{
				loub[j] = rand.NextDouble() * 2 - 1.0;
				for (int i = 0; i < 20; i++)
					louw[j, i] = rand.NextDouble() * 2 - 1.0;
			}
		}

		// #####################################################################
		// ####							Methods								####
		// #####################################################################

		// set to 0 all derivatives
		public void InitDeriv()
		{
			Array.Clear(dl1b, 0, dl1b.Length);
			Array.Clear(dl1w, 0, dl1w.Length);
			Array.Clear(dl2b, 0, dl2b.Length);
			Array.Clear(dl2w, 0, dl2w.Length);
			Array.Clear(dloub, 0, dloub.Length);
			Array.Clear(dlouw, 0, dlouw.Length);
			Array.Clear(dcz, 0, dcz.Length);
			Array.Clear(dcal2, 0, dcal2.Length);
			Array.Clear(dcal1, 0, dcal1.Length);
		}

		// first execute the network with each given set, then compute the
		// relevant derivatives, then applies them to the weights and biases
		// according to the learning rate 
		public void Train(List<Tup> lst, double lr)
		{
			// calculating derivatives
			InitDeriv();
			foreach (var t in lst)
			{
				double[] y = new double[2];
				double curCZ;
				double tmp;

				SetY(ref y, t.mal);
				ExeNR(t.data);

				// output layer
				for (int j = 0; j < 2; j++)
				{
					curCZ = lou[j] - y[j];

					dcz[j] = lou[j] * (1 - lou[j]) * 2 * curCZ;
					dloub[j] += dcz[j];
					for (int i = 0; i < 20; i++)
						dlouw[j, i] += dcz[j] * l2[i];
				}
				// layer 2
				for (int j = 0; j < 20; j++)
				{
					for (int i = 0; i < 2; i++) // here i represent next layer
						dcal2[j] += dcz[i] * louw[i, j];
					curCZ = dcal2[j];

					tmp = l2[j] * (1 - l2[j]) * 2 * curCZ;
					dl2b[j] += tmp;
					for (int i = 0; i < 20; i++)
						dl2w[j, i] += tmp * l1[i];
				}


				// layer 1
				for (int j = 0; j < 20; j++)
				{
					for (int i = 0; i < 20; i++) // here i represent next layer
						dcal1[j] += dcal2[i] * l2w[i, j];
					curCZ = dcal1[j];

					tmp = l1[j] * (1 - l1[j]) * 2 * curCZ;
					dl1b[j] += tmp;
					for (int i = 0; i < 30; i++)
						dl1w[j, i] += tmp * t.data[i];
				}

			}

			// gradient descend
			for (int j = 0; j < 2; j++) // output layer
			{
				for (int i = 0; i < 20; i++)
					louw[j, i] -= dlouw[j, i] * lr;
				loub[j] -= dloub[j] * lr;
			}
			for (int j = 0; j < 20; j++) // layer 2
			{
				for (int i = 0; i < 20; i++)
					l2w[j, i] -= dl2w[j, i] * lr;
				l2b[j] -= dl2b[j] * lr;
			}
			for (int j = 0; j < 20; j++) // layer 1
			{
				for (int i = 0; i < 30; i++)
					l1w[j, i] -= dl1w[j, i] * lr;
				l1b[j] -= dl1b[j] * lr;
			}
		}








		// no return execution, just sets the fields
		private void ExeNR(double[] inputs)
		{
			if (inputs.Length != 30)
				throw new ArgumentException("Inputs should be 30 long");
			for(int neur = 0; neur < 20; neur++)
			{
				l1[neur] = l1b[neur];
				for (int prNeur = 0;  prNeur < 30; prNeur++)
					l1[neur] += inputs[prNeur] * l1w[neur, prNeur];
				l1[neur] = Activ(l1[neur]);
			}
			for(int neur = 0; neur < 20; neur++)
			{
				l2[neur] = l2b[neur];
				for (int prNeur = 0;  prNeur < 20; prNeur++)
					l2[neur] += l1[prNeur] * l2w[neur, prNeur];
				l2[neur] = Activ(l2[neur]);
			}
			for(int neur = 0; neur < 2; neur++)
			{
				lou[neur] = loub[neur];
				for (int prNeur = 0;  prNeur < 20; prNeur++)
					lou[neur] += l2[prNeur] * louw[neur, prNeur];
				lou[neur] = Activ(lou[neur]);
			}
		}
		
		// return a pair of doubles, first benign 2nd malignant
		public double[] Exe(double[] inputs)
		{
			double[] ret = new double[2];

			ExeNR(inputs);
			ret[0] = lou[0];
			ret[1] = lou[1];
			return ret;
		}

		private void SetY(ref double[] y, bool mal)
		{
			if (mal)
			{
				y[0] = 0.0;
				y[1] = 1.0;
			}
			else
			{
				y[0] = 1.0;
				y[1] = 0.0;
			}
		}

		public double Cost(List<Tup> lst)
		{
			double cost = 0.0;
			double tmp;
			double[] y = new double[2];

			foreach (Tup t in lst)
			{
				double[] res = Exe(t.data);
				SetY(ref y, t.mal);
				tmp = res[0] - y[0];
				cost += tmp * tmp;
				tmp = res[1] - y[1];
				cost += tmp * tmp;
			}
			return (cost /= lst.Count);
		}
	}
}