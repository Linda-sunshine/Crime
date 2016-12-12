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
import Regression.MyLinearRegression;

public class MyLRMain {
	//In the main function, we want to input the data and do adaptation 
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{
		try{
			String fv = "seed"; //"fs", "demo"
			String type = "black";// "black" or "gay"
							
			String trainFile = String.format("./data/%s_train_%s.arff", type, fv);		
			String testFile = String.format("./data/%s_test_%s.arff", type, fv);

			LinearRegression lr = new LinearRegression();
			
			System.out.println(String.format("Start loading %s training data....", type));
			BufferedReader trainReader = new BufferedReader(new FileReader(trainFile));
			Instances train = new Instances(trainReader);
			train.setClassIndex(train.numAttributes() - 1);
			lr.buildClassifier(train);
			
			System.out.println(String.format("Start loading %s testing data....", type));
			BufferedReader testReader = new BufferedReader(new FileReader(testFile));
			Instances test = new Instances(testReader);
			test.setClassIndex(test.numAttributes() - 1);

			System.out.println("Start evaluation...");
			Evaluation eval = new Evaluation(train);
			eval.evaluateModel(lr, test);
			System.out.println(eval.toSummaryString("\nResults\n======\n", false));
		} catch(Exception e){
			e.printStackTrace();
		}
	}
}
