package mains;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;

import utils.Utils;
import weka.classifiers.functions.LinearRegression;
import weka.core.Instance;
import weka.core.Instances;

/*** The purpose of this function is to input data and the learned models, 
 * output the corresponding predicted values.
 * Input: features, learned models, test tweet data
 * Process: text processing of the tweets
 * Output: four predicted attitudes data.
 * 
 * @author lin
 *
 */

public class IncidentMain {
	public static double classify(double[] weights, Instance ins){
		double pred = weights[0]; // bias term
		for(int i=0; i<ins.numAttributes(); i++){
			if(ins.value(i) != 0)
				pred += weights[i]*ins.value(i);
		}
		return pred;
	}
	public static void main(String[] args) throws Exception{

		int k = 2000;
		boolean demo = false;
		String type = "black", att = "imp", fv = "df";
		String model = String.format("./data/incident/%s_train_%s_%s_%d_demo_%b.txt", type, att, fv, k, demo);
	
//		String prefix = "/if15/lg5bt/DSIData";//"./data/"
		String prefix = "./data/";
		String data = "geo";

		String trainFile = String.format("%s/%s/ArffData/%s_test_%s_%s_%d_demo_%b.arff", prefix, data, type, att, fv, k, demo);	
		String testFile = String.format("%s/%s/ArffData/%s_test_%s_%s_%d_demo_%b.arff", prefix, data, type, att, fv, k, demo);
		
		try{
			/***Implicit attitudes.****/
			LinearRegression lr = new LinearRegression();
			
			BufferedReader trainReader = new BufferedReader(new FileReader(trainFile));
			Instances train = new Instances(trainReader);
			train.setClassIndex(train.numAttributes() - 1);
			System.out.print("Total number of attributes is "+train.numAttributes());
			lr.buildClassifier(train);
			System.out.println(String.format("Start loading %s testing data from %s....", type, testFile));
			
			BufferedReader testReader = new BufferedReader(new FileReader(testFile));
			Instances test = new Instances(testReader);
			test.setClassIndex(test.numAttributes() - 1);
			double[] coefficients = lr.coefficients();
			
			Instance ins;
			double trueY = 0, predY = 0, dotY = 0 ; 
			for(int i=0; i<test.size(); i++){
				ins = test.get(i);
				trueY = ins.value(test.numAttributes()-1);
				int size = ins.numAttributes();
				System.out.println("Ins attribute number: " + size);
				for(int s=0; s<size; s++){
					System.out.println(s + ins.attribute(s).name() + "\t" + ins.value(s));
				}
				predY = lr.classifyInstance(ins);
				dotY = classify(coefficients, ins);
				System.out.print(String.format("TrueY:%.4f, predY by lr:%.4f, predY by self:%.4f\n", trueY, predY, dotY));
			}
		}catch(IOException e){
			e.printStackTrace();
		}
	}
}
