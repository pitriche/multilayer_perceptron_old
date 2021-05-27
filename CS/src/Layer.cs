using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;


namespace CS
{
	public class Layer
	{
		public int nInput { get; private set; }
		public int nNeuron { get; private set; }

		public double[,] weight;
		public double[] bias;

		private double[] input; // execution time input
		private double[] Z; // execution time weighted sum
		public double[] A; // execution time neuron output (activation function on the weighted sum)

		// derivatives

		/*
		** used to find the derivative of the weights, biases and inputs
		** by multiplicating their respective derivatives with the weighted sum
		** to cost derivative
		*/
		private double[] dAC; // layer output to the cost
		private double[] dZA; // weighted sum to the layer output
		// bias to the weighted sum, dBZ, is 1

		// final derivatives
		public double[,] dWC; // weights to cost
		public double[] dBC; // biases to cost
		public double[] dIC; // input to cost

		// final derivatives averages
		public double[,] adWC; // weights to cost
		public double[] adBC; // biases to cost
		public double[] adIC; // input to cost

		private Func<double, double> Activ; // neuron activation function

		// #####################################################################
		// ####							Constructor							####
		// #####################################################################

		public Layer(int nInput, int nNeuron)
		{
			this.nInput = nInput;
			this.nNeuron = nNeuron;
			
			weight = new double[nNeuron, nInput];
			bias = new double[nNeuron];
			
			Z = new double[nNeuron];
			A = new double[nNeuron];
			
			dZA = new double[nNeuron];
			
			dWC = new double[nNeuron, nInput];
			dBC = new double[nNeuron];
			dIC = new double[nInput];

			adWC = new double[nNeuron, nInput];
			adBC = new double[nNeuron];
			Activ = x => 1 / (1 + Math.Pow(Math.E, -x)); // sigmoid activation func
		}

		// #####################################################################
		// ####							Methods								####
		// #####################################################################

		public void RandWB(Random rand)
		{
			for (int j = 0; j < nNeuron; j++)
			{
				bias[j] = rand.NextDouble() * 2 - 1;
				for (int i = 0; i < nInput; i++)
					weight[j, i] =  rand.NextDouble() * 2 - 1;
			}
		}

		/*
		** compute the weighted sums, and applies the activation function
		** on the results. Needs to be run before Derivatives
		*/
		public double[] Exe(double[] input)
		{
			if (input.Length != nInput)
				throw new Exception("Inputs have to be " + nInput + " long");
			this.input = input;
			Array.Clear(Z, 0, Z.Length);
			for (int j = 0; j < nNeuron; j++)
			{
				for (int i = 0; i < nInput; i++)
					Z[j] += input[i] * weight[j, i];
				Z[j] += bias[j];
				A[j] = Activ(Z[j]);
			}
			return (A);
		}

		public override string ToString()
		{
			StringBuilder sb = new StringBuilder();
			for (int j = 0; j < nNeuron; j++)
			{
				sb.AppendFormat("Neuron {0}:  [bias: {1}]\n", j, bias[j]);
				for (int i = 0; i < nInput; i++)
				{
					sb.AppendFormat("	weight {0}: {1:0.00000}", i, weight[j, i]);
					if ((i + 1) % 4 == 0)
						sb.AppendLine();
				}
				sb.AppendLine();
			}
			return (sb.ToString());
		}

		// #####################################################################
		// ####							Maths								####
		// #####################################################################

		/*
		** For one single training sample
		** calculates the derivatives of the weights, biases and inputs
		** from the derivative of the output to the cost
		*/
		public void Derivatives(double[] dAC)
		{
			// dAc
			// given either by next layer or squared difference of output and y
			this.dAC = dAC;

			// dZA
			// A is Activ(Z) witch is sigmoid, so it's used to compute the
			// derivative of the sigmoid function, which is sig(x) * (1 - sig(x))
			for (int j = 0; j < nNeuron; j++)
				dZA[j] = A[j] * (1 - A[j]);

			// dWZ[j, i] = input[i]

			// dIZ[j, i] = weight[j, i];

			// final derivatives
			// dBC, dWC and dIC
			// due to dBZ being one, bWC is effectively dWZ * dBC;
			Array.Clear(dIC, 0, dIC.Length);
			for (int j = 0; j < nNeuron; j++)
			{
				dBC[j] = dZA[j] * dAC[j];
				for (int i = 0; i < nInput; i++)
				{
					dWC[j, i] = input[i] * dBC[j];
					dIC[i] += weight[j, i] * dBC[j];
				}
			}

			// adding to the averages
			for (int j = 0; j < nNeuron; j++)
			{
				adBC[j] += dBC[j];
				for (int i = 0; i < nInput; i++)
					adWC[j, i] += dWC[j, i];
			}
		}

		// divide the averages sum by the amount of data used
		public void Averages(int count)
		{
			for (int j = 0; j < nNeuron; j++)
			{
				adBC[j] /= count;
				for (int i = 0; i < nInput; i++)
					adWC[j, i] /= count;
			}
		}

		/*
		** applies gradient descend to the weights and biases with the averaged
		** derivatives
		*/
		public void GradientDescend(double lr)
		{

			for (int j = 0; j < nNeuron; j++)
			{
				bias[j] -= adBC[j] * lr;
				for (int i = 0; i < nInput; i++)
					weight[j, i] -= adWC[j, i] * lr;
			}
		}
	}
}
