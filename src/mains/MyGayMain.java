package mains;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import opennlp.tools.util.InvalidFormatException;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.core.Instances;


public class MyGayMain {
	//In the main function, we want to input the data and do adaptation 
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{
		try{
			System.out.println("Start loading gay training data....");
			String trainSeedFile = "./data/gay_train_seed.arff";
			String testSeedFile = "./data/gay_test_seed.arff";

			LinearRegression lr = new LinearRegression();

			BufferedReader trainReader = new BufferedReader(new FileReader(trainSeedFile));
			Instances train = new Instances(trainReader);
			train.setClassIndex(train.numAttributes() - 1);
			lr.buildClassifier(train);
			
			System.out.println("Start loading gay testing data....");
			BufferedReader testReader = new BufferedReader(new FileReader(testSeedFile));
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
