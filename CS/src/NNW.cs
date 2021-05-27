using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;


namespace CS
{
	/*
	** Mudular neural network class (v1.0)
	*/
	public class NNW
	{
		private Layer[] L;
		private int nInput;
		private int nOutput;
		private int nLayers;

		public Model model;

		// #####################################################################
		// ####							Constructors						####
		// #####################################################################

		// layersSize mentions the sizes of all layers from input to output
		// parameters aren't sanitized, and that's a choice
		public NNW(int seed, int[] layersSize)
		{
			Random rand = new Random(seed);

			model = new Model(layersSize);
			nLayers = layersSize.Length - 1;
			nInput = layersSize[0];
			nOutput = layersSize[nLayers];
			L = new Layer[nLayers];
			for (int i = 0; i < nLayers; i++)
			{
				L[i] = new Layer(layersSize[i], layersSize[i + 1]);
				L[i].RandWB(rand);
			}
		}

		// create network from model
		public NNW(Model model)
		{
			this.model = model;
			nLayers = model.layersSize.Length - 1;
			nInput = model.layersSize[0];
			nOutput = model.layersSize[nLayers];
			L = new Layer[nLayers];
			for (int i = 0; i < nLayers; i++)
				L[i] = new Layer(model.layersSize[i], model.layersSize[i + 1]);
			UnpackModel();
		}

		// #####################################################################
		// ####							Methods								####
		// #####################################################################

		public double[] Exe(double[] input)
		{
			double[] output = input;
			for (int i = 0; i < nLayers; i++)
				output = L[i].Exe(output);
			return (output);
		}

		/*
		** calculate and average the derivatives, then applies gradient descend
		** on the weights and biases
		*/
		public void Train(List<Tup> lst, double lr)
		{
			double[] dAC = new double[nOutput];
			// init all averages to 0
			foreach (var lay in L)
			{
				Array.Clear(lay.adBC, 0, lay.adBC.Length);
				Array.Clear(lay.adWC, 0, lay.adWC.Length);
			}

			// for each training tuple in the set
			foreach(var t in lst)
			{
				// execute the network and compute the output to cost derivative
				double[] A = this.Exe(t.data);
				for (int j = 0; j < nOutput; j++)
					dAC[j] = 2 * (A[j] - t.y);
				// compute each layer's derivatives with the next's input to cost derivative
				L[2].Derivatives(dAC);
				L[1].Derivatives(L[2].dIC);
				L[0].Derivatives(L[1].dIC);
			}

			L[2].Averages(lst.Count);
			L[1].Averages(lst.Count);
			L[0].Averages(lst.Count);

			L[2].GradientDescend(lr);
			L[1].GradientDescend(lr);
			L[0].GradientDescend(lr);
		}

		public double Cost(List<Tup> lst)
		{
			double cost = 0.0;
			double tmp;

			foreach(var t in lst)
			{
				double[] res = this.Exe(t.data);
				tmp = res[0] - t.y;
				cost += tmp * tmp;
			}
			return (cost / lst.Count);
		}

		public override string ToString()
		{
			StringBuilder sb = new StringBuilder();

			sb.AppendFormat("Input layer: {0} inputs\n\n", nInput);
			for (int i = 0; i < nLayers; i++)
			{
				sb.AppendFormat("Layer {0} : {1} neurons #########################\n"
					, i + 1, L[i].nNeuron);
				sb.Append(L[i].ToString());
			}
			return (sb.ToString());
		}

		public void PackModel()
		{
			int im = 0;
			foreach (Layer layer in L)
			{
				for (int j = 0; j < layer.nNeuron; j++)
				{
					for (int i = 0; i < layer.nInput; i++)
						model.weightsBiases[im++] = layer.weight[j, i];
					model.weightsBiases[im++] = layer.bias[j];
				}
			}
		}

		public void UnpackModel()
		{
			int im = 0;

			for (int il = 0; il < nLayers; il++)
			{
				Layer layer = L[il];
				for (int j = 0; j < layer.nNeuron; j++)
				{
					for (int i = 0; i < layer.nInput; i++)
						layer.weight[j, i] = model.weightsBiases[im++];
					layer.bias[j] = model.weightsBiases[im++];
				}
			}
		}
	}
	
}
