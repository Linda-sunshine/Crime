package mains;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

import opennlp.tools.util.InvalidFormatException;
import structures._Doc;
import structures._User;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
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
			int[] ks = new int[]{2000, 3000, 4000};
			for(int k: ks){
			String fv = "df"; //"df", "demo"
			String type = "gay";// "black" or "gay"
							
			String trainImpFile = String.format("/if15/lg5bt/ArffData/%s_train_imp_%s_%d.arff", type, fv, k);	
			String trainExpFile = String.format("/if15/lg5bt/ArffData/%s_train_exp_%s_%d.arff", type, fv, k);
			String testImpFile = String.format("/if15/lg5bt/ArffData/%s_test_imp_%s_%d.arff", type, fv, k);
			String testExpFile = String.format("/if15/lg5bt/ArffData/%s_test_exp_%s_%d.arff", type, fv, k);

			LinearRegression lr = new LinearRegression();
			
			System.out.println(String.format("Start loading %s training data from %s....", type, trainImpFile));
			BufferedReader trainReader = new BufferedReader(new FileReader(trainImpFile));
			Instances train = new Instances(trainReader);
			train.setClassIndex(train.numAttributes() - 1);
			lr.buildClassifier(train);
			
			System.out.println(String.format("Start loading %s testing data from %s....", type, testImpFile));
			BufferedReader testReader = new BufferedReader(new FileReader(testImpFile));
			Instances test = new Instances(testReader);
			test.setClassIndex(test.numAttributes() - 1);

			System.out.println("Start evaluation...");
			Evaluation eval = new Evaluation(train);
			eval.evaluateModel(lr, test);
			System.out.println(eval.toSummaryString("\nResults For Implicit Attitudes\n======\n", false));
			
			System.out.println(String.format("Start loading %s training data from %s....", type, trainExpFile));
			trainReader = new BufferedReader(new FileReader(trainExpFile));
			train = new Instances(trainReader);
			train.setClassIndex(train.numAttributes() - 1);
			lr.buildClassifier(train);
			
			System.out.println(String.format("Start loading %s testing data from %s....", type, testExpFile));
			testReader = new BufferedReader(new FileReader(testExpFile));
			test = new Instances(testReader);
			test.setClassIndex(test.numAttributes() - 1);

			System.out.println("Start evaluation...");
			eval = new Evaluation(train);
			eval.evaluateModel(lr, test);
			System.out.println(eval.toSummaryString("\nResults for Explicit Results\n======\n", false));
			
			}
		} catch(Exception e){
			e.printStackTrace();
		}
	}
}
