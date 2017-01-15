package mains;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;

import opennlp.tools.util.InvalidFormatException;
import structures.MyPriorityQueue;
import structures._Doc;
import structures._RankItem;
import structures._User;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.core.Attribute;
import weka.core.Instances;
import Analyzer.Analyzer;
import Analyzer.CrossFeatureSelection;
import Analyzer.MultiThreadedUserAnalyzer;
import Analyzer.UserAnalyzer;
import Classifier.supervised.GlobalSVM;
import Classifier.supervised.SVM;
import Classifier.supervised.modelAdaptation.HDP.MTCLRWithHDP;

public class MyLRMain {
	
	//In the main function, we want to input the data and do adaptation 
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{
		try{
			int[] ks = new int[]{2000};
			for(int k: ks){
			String fv = "df"; //"df", "demo"
			String type = "gay";// "black" or "gay"
		
//			String trainImpFile = String.format("/if15/lg5bt/ArffData/%s_train_imp_%s_%d.arff", type, fv, k);	
//			String trainExpFile = String.format("/if15/lg5bt/ArffData/%s_train_exp_%s_%d.arff", type, fv, k);
//			String testImpFile = String.format("/if15/lg5bt/ArffData/%s_test_imp_%s_%d.arff", type, fv, k);
//			String testExpFile = String.format("/if15/lg5bt/ArffData/%s_test_exp_%s_%d.arff", type, fv, k);

			String trainImpFile = String.format("./data/ArffData/%s_train_imp_%s_%d.arff", type, fv, k);	
			String trainExpFile = String.format("./data/ArffData/%s_train_exp_%s_%d.arff", type, fv, k);
			String testImpFile = String.format("./data/ArffData/%s_test_imp_%s_%d.arff", type, fv, k);
			String testExpFile = String.format("./data/ArffData/%s_test_exp_%s_%d.arff", type, fv, k);
			
			LinearRegression lr = new LinearRegression();
			
			System.out.println(String.format("Start loading %s training data from %s....", type, trainImpFile));
			BufferedReader trainReader = new BufferedReader(new FileReader(trainImpFile));
			Instances train = new Instances(trainReader);
			train.setClassIndex(train.numAttributes() - 1);
			lr.buildClassifier(train);
			
			int topK = 100;		
			// Sort the weights of the learned features.
			double[] weights = lr.coefficients();
			MyPriorityQueue<_RankItem> rankq = new MyPriorityQueue<_RankItem>(topK);
			for(int i=0; i<weights.length; i++){
				rankq.add(new _RankItem(i, weights[i]));
			}
			String[] topFvs = new String[topK];
			int in = 0;
			for(_RankItem it: rankq){
				topFvs[in++] = train.attribute(it.m_index).name();
			}
			
			try{
				PrintWriter writer = new PrintWriter(new File(String.format("./data/%s_imp_top_%d.txt", type, topK)));
				for(String f: topFvs)
					writer.write(f+"\n");
				writer.close();
			} catch(IOException e){
				e.printStackTrace();
			}			
			
//			System.out.println(String.format("Start loading %s testing data from %s....", type, testImpFile));
//			BufferedReader testReader = new BufferedReader(new FileReader(testImpFile));
//			Instances test = new Instances(testReader);
//			test.setClassIndex(test.numAttributes() - 1);

//			System.out.println("Start evaluation...");
//			Evaluation eval = new Evaluation(train);
//			eval.evaluateModel(lr, test);
//			System.out.println(eval.toSummaryString("\nResults For Implicit Attitudes\n======\n", false));
			
			System.out.println(String.format("Start loading %s training data from %s....", type, trainExpFile));
			trainReader = new BufferedReader(new FileReader(trainExpFile));
			train = new Instances(trainReader);
			train.setClassIndex(train.numAttributes() - 1);
			lr.buildClassifier(train);
			
//			System.out.println(String.format("Start loading %s testing data from %s....", type, testExpFile));
//			testReader = new BufferedReader(new FileReader(testExpFile));
//			test = new Instances(testReader);
//			test.setClassIndex(test.numAttributes() - 1);
//
//			System.out.println("Start evaluation...");
//			eval = new Evaluation(train);
//			eval.evaluateModel(lr, test);
//			System.out.println(eval.toSummaryString("\nResults for Explicit Results\n======\n", false));
			
			
			// Sort the weights of the learned features.
			weights = lr.coefficients();
			rankq = new MyPriorityQueue<_RankItem>(topK);
			for(int i=0; i<weights.length; i++){
				rankq.add(new _RankItem(i, weights[i]));
			}
			topFvs = new String[topK];
			in = 0;
			for(_RankItem it: rankq){
				topFvs[in++] = train.attribute(it.m_index).name();
			}
			
			try{
				PrintWriter writer = new PrintWriter(new File(String.format("./data/%s_exp_top_%d.txt", type, topK)));
				for(String f: topFvs)
					writer.write(f+"\n");
				writer.close();
			} catch(IOException e){
				e.printStackTrace();
			}
			}
		} catch(Exception e){
			e.printStackTrace();
		}
	}
}

