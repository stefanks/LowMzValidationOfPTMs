using Accord.MachineLearning.VectorMachines;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.Statistics.Analysis;
using Accord.Statistics.Kernels;
using IO.MzML;
using LibSVMsharp;
using LibSVMsharp.Helpers;
using MassSpectrometry;
using MathNet.Numerics.Statistics;
using Spectra;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LowMzValidationOfPTMs
{
    class Program
    {
        static void Main(string[] args)
        {
            //List<int> haveMod = new List<int>();
            //List<int> doNotHaveMod = new List<int>();
            //Identifications identifications = new MzidIdentifications(@"C:\Users\stepa\Data\jurkat\Original\120426_Jurkat_highLC_Frac6-Calibrated.mzid");
            //var yah = new Mzml(@"C:\Users\stepa\Data\jurkat\Original\120426_Jurkat_highLC_Frac6-Calibrated.mzML");
            //yah.Open();
            //for (int i = 0; i < identifications.Count(); i++)
            //{
            //    if (i > 6099)
            //        break;
            //    for (int j = 0; j < identifications.NumModifications(i); j++)
            //    {
            //        if (identifications.modificationAcession(i, j).Equals("PSI-MOD:00046"))
            //        {
            //            haveMod.Add(i);
            //            break;
            //        }
            //    }
            //    if (haveMod.Count == 0 || haveMod.Last() != i)
            //        doNotHaveMod.Add(i);
            //}
            //Console.WriteLine("haveMod.Count: " + haveMod.Count);
            //Console.WriteLine("doNotHaveMod.Count: " + doNotHaveMod.Count);

            //var a = new Random(1);
            //haveMod.Shuffle(a);
            //doNotHaveMod.Shuffle(a);

            //var trainingHaveMod = haveMod.Take(haveMod.Count * 3 / 4);
            //var trainingDoNotHaveMod = doNotHaveMod.Take(doNotHaveMod.Count * 3 / 4);
            //var testingHaveMod = haveMod.Skip(haveMod.Count * 3 / 4);
            //var testingDoNotHaveMod = doNotHaveMod.Skip(doNotHaveMod.Count * 3 / 4);

            //// WRITE TRAINING AND TESTING TO FILES!

            //double[][] inputs = new double[trainingHaveMod.Count() + trainingDoNotHaveMod.Count()][];
            //int ok = 32768;
            //for (int i = 0; i < trainingHaveMod.Count() + trainingDoNotHaveMod.Count(); i++)
            //    inputs[i] = extractFeatures(i < trainingHaveMod.Count() ? identifications.ms2spectrumIndex(trainingHaveMod.ElementAt(i)) : identifications.ms2spectrumIndex(trainingDoNotHaveMod.ElementAt(i - trainingHaveMod.Count())), yah, ok);

            //Console.WriteLine("writing training...");
            //using (StreamWriter file = new StreamWriter(@"training.txt"))
            //{
            //    for (int i = 0; i < trainingHaveMod.Count(); i++)
            //    {
            //        file.Write("+1");
            //        for (int j = 0; j < inputs[i].Count(); j++)
            //        {
            //            if (inputs[i][j] == 1)
            //                file.Write(" " + (j + 1) + ":1");
            //        }
            //        file.WriteLine();
            //    }
            //    int notHaveMod = trainingDoNotHaveMod.Count();
            //    int haveModd = trainingHaveMod.Count();
            //    for (int i = 0; i < notHaveMod; i++)
            //    {
            //        file.Write("-1");
            //        for (int j = 0; j < inputs[i + haveModd].Count(); j++)
            //        {
            //            if (inputs[i + haveModd][j] == 1)
            //                file.Write(" " + (j + 1) + ":1");
            //        }
            //        file.WriteLine();
            //    }
            //}




            //inputs = new double[testingHaveMod.Count() + testingDoNotHaveMod.Count()][];
            //for (int i = 0; i < testingHaveMod.Count() + testingDoNotHaveMod.Count(); i++)
            //    inputs[i] = extractFeatures(i < testingHaveMod.Count() ? identifications.ms2spectrumIndex(testingHaveMod.ElementAt(i)) : identifications.ms2spectrumIndex(testingDoNotHaveMod.ElementAt(i - testingHaveMod.Count())), yah, ok);

            //Console.WriteLine("writing testing...");
            //using (StreamWriter file = new StreamWriter(@"testing.txt"))
            //{
            //    for (int i = 0; i < testingHaveMod.Count(); i++)
            //    {
            //        file.Write("+1");
            //        for (int j = 0; j < inputs[i].Count(); j++)
            //        {
            //            if (inputs[i][j] == 1)
            //                file.Write(" " + (j + 1) + ":1");
            //        }
            //        file.WriteLine();
            //    }
            //    int notHaveMod = testingDoNotHaveMod.Count();
            //    int haveModd = testingHaveMod.Count();
            //    for (int i = 0; i < notHaveMod; i++)
            //    {
            //        file.Write("-1");
            //        for (int j = 0; j < inputs[i + haveModd].Count(); j++)
            //        {
            //            if (inputs[i + haveModd][j] == 1)
            //                file.Write(" " + (j + 1) + ":1");
            //        }
            //        file.WriteLine();
            //    }
            //}


            SVMProblem problem = SVMProblemHelper.Load(@"training.txt");
            SVMProblem testProblem = SVMProblemHelper.Load(@"testing.txt");

            SVMParameter parameter = new SVMParameter();
            parameter.Type = SVMType.C_SVC;
            parameter.Kernel = SVMKernelType.LINEAR;

            double[] importance = new double[32768];
            for (int i = 0; i < importance.Length; i++)
                importance[i] = 1;

            while (true)
            {
                SVMProblem problemInner = new SVMProblem();
                int zerosPercent = importance.Count((b) => b == 0) * 100 / importance.Count();
                double threshold = importance.Percentile(zerosPercent + 5);
                Console.WriteLine(threshold);
                for (int i = 0; i < problem.Length; i++)
                {
                    SVMNode[] x = extractX(problem.X[i], importance, threshold);
                    problemInner.Add(x, problem.Y[i]);
                }
                SVMProblem testProblemInner = new SVMProblem();
                for (int i = 0; i < testProblem.Length; i++)
                {
                    SVMNode[] x = extractX(testProblem.X[i], importance, threshold);
                    testProblemInner.Add(x, testProblem.Y[i]);
                }

                SVMModel model = SVM.Train(problemInner, parameter);

                Console.WriteLine("done training");
                Console.WriteLine("predicting:");

                double[] target = new double[testProblemInner.Length];
                for (int i = 0; i < testProblemInner.Length; i++)
                    target[i] = SVM.Predict(model, testProblemInner.X[i]);

                Console.WriteLine("done predicting:");

                //double accuracy = SVMHelper.EvaluateClassificationProblem(testProblem, target);

                //Console.WriteLine("accuracy:" + accuracy);

                int falseNeg = 0;
                int falsePos = 0;
                int trueNeg = 0;
                int truePos = 0;
                for (int i = 0; i < target.Count(); i++)
                {
                    var targeti = target[i];
                    var testProblemYi = testProblemInner.Y[i];
                    if (targeti > 0 && testProblemYi > 0)
                        truePos++;
                    else if (targeti > 0 && testProblemYi < 0)
                        falsePos++;
                    else if (targeti < 0 && testProblemYi > 0)
                        falseNeg++;
                    else if (targeti < 0 && testProblemYi < 0)
                        trueNeg++;

                }
                Console.WriteLine("falseNeg = " + falseNeg);
                Console.WriteLine("falsePos = " + falsePos);
                Console.WriteLine("trueNeg = " + trueNeg);
                Console.WriteLine("truePos = " + truePos);

                Console.WriteLine("MCC = " + ((double)truePos * trueNeg - falsePos * falseNeg) / (Math.Sqrt(((double)truePos + falsePos) * (truePos + falseNeg) * (trueNeg + falsePos) * (trueNeg * falseNeg))));


                Console.WriteLine("model.ClassCount = " + model.ClassCount);
                Console.WriteLine("model.Creation = " + model.Creation);
                Console.WriteLine("model.TotalSVCount = " + model.TotalSVCount);
                Console.WriteLine("model.Labels = " + string.Join(",", model.Labels.Take(3)) + "...");
                Console.WriteLine("model.SVCounts = " + string.Join(",", model.SVCounts.Take(3)) + "...");
                Console.WriteLine("model.SVCoefs[0] = " + string.Join(",", model.SVCoefs[0].Take(3)) + "...");


                importance = new double[32768];
                for (int i = 0; i < model.SV.Count(); i++)
                {
                    var sv = model.SV[i];
                    foreach (var svEntry in sv)
                    {
                        importance[svEntry.Index] += model.SVCoefs[0][i] * svEntry.Value;
                    }
                }
                Console.WriteLine("importance.Count((b)=>b==0) = " + importance.Count((b) => b == 0));
                Console.WriteLine("importance.Count((b)=>b!=0) = " + importance.Count((b) => b != 0));

                using (StreamWriter file = new StreamWriter(importance.Count((b) => b != 0) + ".txt"))
                {
                    for (int i = 0; i < importance.Count(); i++)
                    {
                        importance[i] = Math.Pow(importance[i], 2);
                        file.WriteLine(importance[i]);
                    }
                }

                //Console.WriteLine("writing...");
                //using (System.IO.StreamWriter file = new System.IO.StreamWriter(@"intervals.txt"))
                //{
                //    for (int i = 0; i < machine.SupportVectors[0].Count(); i++)
                //    {
                //        double sum = 0;
                //        for (int j = 0; j < machine.Weights.Length; j++)
                //        {
                //            sum += machine.Weights[j] * machine.SupportVectors[j][i];
                //        }
                //        file.WriteLine(sum);
                //    }


                //}
            }








            //parameter.Type = SVMType.C_SVC;
            //parameter.Kernel = SVMKernelType.LINEAR;

            //model = SVM.Train(problem, parameter);

            //target = new double[testProblem.Length];
            //for (int i = 0; i < testProblem.Length; i++)
            //    target[i] = SVM.Predict(model, testProblem.X[i]);

            //accuracy = SVMHelper.EvaluateClassificationProblem(testProblem, target);

            //Console.WriteLine("accuracy:" + accuracy);



            //falseNeg = 0;
            //falsePos = 0;
            //trueNeg = 0;
            //truePos = 0;
            //for (int i = 0; i < target.Count(); i++)
            //{
            //    if (target[i] > 0 && testProblem.Y[i] > 0)
            //        truePos++;
            //    if (target[i] > 0 && testProblem.Y[i] < 0)
            //        falsePos++;
            //    if (target[i] < 0 && testProblem.Y[i] > 0)
            //        falseNeg++;
            //    if (target[i] < 0 && testProblem.Y[i] < 0)
            //        trueNeg++;

            //}
            //Console.WriteLine("falseNeg = " + falseNeg);
            //Console.WriteLine("falsePos = " + falsePos);
            //Console.WriteLine("trueNeg = " + trueNeg);
            //Console.WriteLine("truePos = " + truePos);

            //Console.WriteLine("MCC = " + ((double)truePos * trueNeg - falsePos * falseNeg) / (Math.Sqrt(((double)truePos + falsePos) * (truePos + falseNeg) * (trueNeg + falsePos) * (trueNeg * falseNeg))));







            //Console.WriteLine("trainingHaveMod.Count: " + trainingHaveMod.Count());
            //Console.WriteLine("trainingDoNotHaveMod.Count: " + trainingDoNotHaveMod.Count());
            //Console.WriteLine("testingHaveMod.Count: " + testingHaveMod.Count());
            //Console.WriteLine("testingDoNotHaveMod.Count: " + testingDoNotHaveMod.Count());

            //int[] output = new int[trainingHaveMod.Count() + trainingDoNotHaveMod.Count()];
            //for (int i = 0; i < trainingHaveMod.Count() + trainingDoNotHaveMod.Count(); i++)
            //    output[i] = i < trainingHaveMod.Count() ? 1 : -1;

            ////double[][] inputs = new double[trainingHaveMod.Count() + trainingDoNotHaveMod.Count()][];

            ////for (int i = 0; i < trainingHaveMod.Count() + trainingDoNotHaveMod.Count(); i++)
            ////    inputs[i] = extract203Features(i < trainingHaveMod.Count() ? identifications.ms2spectrumIndex(trainingHaveMod.ElementAt(i)) : identifications.ms2spectrumIndex(trainingDoNotHaveMod.ElementAt(i - trainingHaveMod.Count())), yah);

            ////var regression = new LogisticRegressionAnalysis(inputs, output.Select((b) => (b + 1) / 2.0).ToArray());
            ////regression.Compute(); // compute the analysis.
            ////computeTesting(testingHaveMod, testingDoNotHaveMod, identifications,(b)=> regression.Regression.Compute(extract203Features(b, yah)) *2-1);
            ////int ok = 32768;

            ////for (int ok = 1; ; ok *= 2)
            ////{
            ////    Console.WriteLine("ok = " + ok);

            ////for (double sensitivity = 1.0; ; sensitivity /= 2)
            ////{
            //double sensitivity = 0.0005;
            //Console.WriteLine("sensitivity = " + sensitivity);

            //List<int> listOfGoodIntevals = loadListFromFile(@"32768.txt", sensitivity);

            //Console.WriteLine("listOfGoodIntevals.Count = " + listOfGoodIntevals.Count);
            //Console.WriteLine("listOfGoodIntevals.Sum = " + listOfGoodIntevals.Sum());

            //for (int i = 0; i < trainingHaveMod.Count() + trainingDoNotHaveMod.Count(); i++)
            //    //    inputs[i] = extractFeaturesIntervals(i < trainingHaveMod.Count() ? identifications.ms2spectrumIndex(trainingHaveMod.ElementAt(i)) : identifications.ms2spectrumIndex(trainingDoNotHaveMod.ElementAt(i - trainingHaveMod.Count())), yah);
            //    inputs[i] = extractFeaturesWithHelpOfFile(i < trainingHaveMod.Count() ? identifications.ms2spectrumIndex(trainingHaveMod.ElementAt(i)) : identifications.ms2spectrumIndex(trainingDoNotHaveMod.ElementAt(i - trainingHaveMod.Count())), yah, listOfGoodIntevals);
            ////inputs[i] = extractFeatures(i < trainingHaveMod.Count() ? identifications.ms2spectrumIndex(trainingHaveMod.ElementAt(i)) : identifications.ms2spectrumIndex(trainingDoNotHaveMod.ElementAt(i - trainingHaveMod.Count())), yah, ok);

            //KernelSupportVectorMachine machine = new KernelSupportVectorMachine(new Linear(), inputs[0].Length);
            //SequentialMinimalOptimization smo = new SequentialMinimalOptimization(machine, inputs, output);
            //smo.UseComplexityHeuristic = false;
            //smo.UseClassProportions = true;
            //smo.Run();


            //Console.WriteLine("writing...");
            //using (System.IO.StreamWriter file = new System.IO.StreamWriter(@"intervals.txt"))
            //{
            //    for (int i = 0; i < machine.SupportVectors[0].Count(); i++)
            //    {
            //        double sum = 0;
            //        for (int j = 0; j < machine.Weights.Length; j++)
            //        {
            //            sum += machine.Weights[j] * machine.SupportVectors[j][i];
            //        }
            //        file.WriteLine(sum);
            //    }
            //}

            ////computeTesting(testingHaveMod, testingDoNotHaveMod, identifications, (s) => machine.Compute(extractFeatures(s, yah, ok)));

            //computeTesting(testingHaveMod, testingDoNotHaveMod, identifications, (s) => machine.Compute(extractFeaturesWithHelpOfFile(s, yah, listOfGoodIntevals)));
            ////computeTesting(testingHaveMod, testingDoNotHaveMod, identifications, (s) => machine.Compute(extractFeaturesIntervals(s, yah)));

            ////}

            Console.Read();

        }

        private static SVMNode[] extractX(SVMNode[] sVMNode, double[] importance, double threshold)
        {
            List<SVMNode> ok = new List<SVMNode>();
            foreach (var a in sVMNode)
            {
                //if (importance[a.Index] > 2.5e-7)
                if (importance[a.Index] >= threshold)
                {
                    ok.Add(a);
                }
                else
                {
                    //    Console.WriteLine("removing " + a.Index + ":" + a.Value + " becuase importance is " + importance[a.Index]);
                }
            }
            return ok.ToArray();
        }

        private static double[] extractFeaturesIntervals(int spectrumIndex, Mzml yah)
        {
            var hehehe = yah.GetScan(spectrumIndex);
            var hm = hehehe.MassSpectrum;
            double[] features = new double[16];

            bool a = hm.ContainsAnyPeaksWithinRange(1546.9, 1549.1);
            bool b = hm.ContainsAnyPeaksWithinRange(1622.3, 1625.2);
            bool c = hm.ContainsAnyPeaksWithinRange(1690, 1697);
            bool d = hm.ContainsAnyPeaksWithinRange(1996.9, 1997.4);
            if (a)
                features[0] = 1;
            else
                features[0] = 0;
            if (b)
                features[1] = 1;
            else
                features[1] = 0;
            if (c)
                features[2] = 1;
            else
                features[2] = 0;
            if (d)
                features[3] = 1;
            else
                features[3] = 0;


            if (a && b)
                features[4] = 1;
            else
                features[4] = 0;
            if (a && c)
                features[5] = 1;
            else
                features[5] = 0;
            if (a && d)
                features[6] = 1;
            else
                features[6] = 0;
            if (b && c)
                features[7] = 1;
            else
                features[7] = 0;
            if (b && d)
                features[8] = 1;
            else
                features[8] = 0;
            if (c && d)
                features[9] = 1;
            else
                features[9] = 0;


            if (a && b && c)
                features[10] = 1;
            else
                features[10] = 0;
            if (a && b && d)
                features[11] = 1;
            else
                features[11] = 0;
            if (b && c && d)
                features[12] = 1;
            else
                features[12] = 0;
            if (a && c && d)
                features[13] = 1;
            else
                features[13] = 0;

            if (a && b && c && d)
                features[14] = 1;
            else
                features[14] = 0;

            if (!a && !b && !c && !d)
                features[15] = 1;
            else
                features[15] = 0;


            return features;
        }

        private static List<int> loadListFromFile(string v, double sensitivity)
        {
            string line;
            int counter = 0;
            List<int> a = new List<int>();

            StreamReader file =
                new StreamReader(v);
            while ((line = file.ReadLine()) != null)
            {
                //Console.WriteLine(counter+":");
                //Console.WriteLine(line);
                //Console.WriteLine(Math.Abs(double.Parse(line)));
                if (Math.Abs(double.Parse(line)) >= sensitivity)
                    a.Add(1);
                else
                {
                    a.Add(0);
                }
                counter++;
            }

            file.Close();
            return a;
        }

        private static double[] extractFeaturesWithHelpOfFile(int spectrumIndex, Mzml yah, List<int> listOfGoodIntevals)
        {
            int ok = 32768;
            var hehehe = yah.GetScan(spectrumIndex);
            var hm = hehehe.MassSpectrum;
            List<double> features = new List<double>();
            double minMZ = 100;
            double maxMZ = 2000;
            for (int i = 0; i < ok; i++)
            {
                if (listOfGoodIntevals[i] == 1)
                {
                    if (hm.ContainsAnyPeaksWithinRange(minMZ + i * (maxMZ - minMZ) / ok, minMZ + (i + 1) * (maxMZ - minMZ) / ok))
                        features.Add(1);
                    else
                        features.Add(-1);
                }
            }
            return features.ToArray();
        }

        private static double[] extractFeatures(int spectrumIndex, Mzml yah, int ok)
        {
            var hehehe = yah.GetScan(spectrumIndex);
            var hm = hehehe.MassSpectrum;
            double[] features = new double[ok];
            double minMZ = 100;
            double maxMZ = 2000;
            for (int i = 0; i < ok; i++)
            {
                if (hm.ContainsAnyPeaksWithinRange(minMZ + i * (maxMZ - minMZ) / ok, minMZ + (i + 1) * (maxMZ - minMZ) / ok))
                    features[i] = 1;
                else
                    features[i] = -1;
            }

            int chargeState;
            hehehe.TryGetSelectedIonGuessChargeStateGuess(out chargeState);
            return features;
        }

        private static void computeTesting(IEnumerable<int> testingHaveMod, IEnumerable<int> testingDoNotHaveMod, Identifications identifications, Func<int, double> predictor)
        {
            int falseNeg = 0;
            int falsePos = 0;
            int trueNeg = 0;
            int truePos = 0;
            for (int i = 0; i < testingHaveMod.Count() + testingDoNotHaveMod.Count(); i++)
            {
                int spectrumIndex = i < testingHaveMod.Count() ? identifications.ms2spectrumIndex(testingHaveMod.ElementAt(i)) : identifications.ms2spectrumIndex(testingDoNotHaveMod.ElementAt(i - testingHaveMod.Count()));
                double y = predictor(spectrumIndex);
                if (i < testingHaveMod.Count() && y > 0)
                    truePos++;
                if (i >= testingHaveMod.Count() && y > 0)
                    falsePos++;
                if (i < testingHaveMod.Count() && y <= 0)
                    falseNeg++;
                if (i >= testingHaveMod.Count() && y <= 0)
                    trueNeg++;

            }
            Console.WriteLine("falseNeg = " + falseNeg);
            Console.WriteLine("falsePos = " + falsePos);
            Console.WriteLine("trueNeg = " + trueNeg);
            Console.WriteLine("truePos = " + truePos);

            Console.WriteLine("MCC = " + ((double)truePos * trueNeg - falsePos * falseNeg) / (Math.Sqrt(((double)truePos + falsePos) * (truePos + falseNeg) * (trueNeg + falsePos) * (trueNeg * falseNeg))));

        }
        private static double[] extract2003Features(int spectrumIndex, IMsDataFile<IMzSpectrum<MzPeak>> yah)
        {
            var hehehe = yah.GetScan(spectrumIndex);
            var hm = hehehe.MassSpectrum;
            double[] features = new double[2003];
            for (int i = 0; i < 2000; i++)
            {
                if (hm.ContainsAnyPeaksWithinRange(i, (i + 1)))
                    features[i] = 1;
                else
                    features[i] = -1;
            }

            int chargeState;
            hehehe.TryGetSelectedIonGuessChargeStateGuess(out chargeState);
            //features[2000] = chargeState;
            //features[2001] = hm.Count;
            //features[2002] = hm.GetSumOfAllY();
            return features;
        }

        private static double[] extract203Features(int spectrumIndex, IMsDataFile<IMzSpectrum<MzPeak>> yah)
        {
            var hehehe = yah.GetScan(spectrumIndex);
            var hm = hehehe.MassSpectrum;
            double[] features = new double[203];
            for (int i = 0; i < 200; i++)
            {
                if (hm.ContainsAnyPeaksWithinRange(i * 10, (i + 1) * 10))
                    features[i] = 1;
                else
                    features[i] = -1;
            }

            int chargeState;
            hehehe.TryGetSelectedIonGuessChargeStateGuess(out chargeState);
            //features[200] = chargeState;
            //features[201] = hm.Count;
            //features[202] = hm.GetSumOfAllY();
            return features;
        }


    }
}
