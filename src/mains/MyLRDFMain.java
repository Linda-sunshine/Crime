package mains;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.Normalizer;
import java.util.ArrayList;

import opennlp.tools.tokenize.Tokenizer;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.util.InvalidFormatException;
import structures.MyPriorityQueue;
import structures._RankItem;
import utils.Utils;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.core.Instances;
import Analyzer.SplitUserAnalyzer;
import Analyzer.UserAnalyzer;

public class MyLRDFMain {

	//In the main function, we want to input the data and do adaptation 
	public static void main(String[] qqqqqqqqgs) throws InvalidFormatException, FileNotFoundException, IOException{
		
		int classNumber = 2;
		int Ngram = 2; // The default value is unigram.
		int lengthThreshold = 0; // Document length threshold

		String tokenModel = "./data/Model/en-token.bin"; // Token model.
		String stopwords = "./data/Model/stopwords.dat";
		
		int k = 4000;
//		String prefix = "/if15/lg5bt/DSIData";//"./data"
		String prefix = "../Crime/data";
		String data = "split";
		String fv = "toplr";
		String type = "black";// "black" or "gay"
		String suffix = ".csv";
		boolean demo = false;// whether we include the demo in the training.
		String att = "exp";
		for(int i=0; i<10; i++){
		String tweetTrain = String.format("%s/%s/splitTweets/tweets_%d", prefix, data, i);
//		String tweetTest = String.format("%s/%s/tweetSplit/tweetsTest/", prefix, data);
		
		String features = String.format("%s/%s/%s_%s_%d_%s.txt", prefix, data, type, fv, k, att);
//		String expFeatures = String.format("%s/%s/%s_%s_%d_imp.txt", prefix, data, type, fv, k);

		String trainIAT = String.format("%s/%s/%sIAT.csv", prefix, data, type);
//		String testIAT = String.format("%s/%s/%sTestIAT.csv", prefix, data, type);
			
		String trainFile = String.format("%s/%s/ArffData/%s_train_imp_%s_%d_demo_%b_%d.arff", prefix, data, type, fv, k, demo, i);	
//		String trainExpFile = String.format("%s/%s/ArffData/%s_train_exp_%s_%d_demo_%b_%d.arff",prefix, data, type, fv, k, demo, i);
//		String testImpFile = String.format("%s/%s/ArffData/%s_test_imp_%s_%d_demo_%b.arff", prefix, data, type, fv, k, demo);
//		String testExpFile = String.format("%s/%s/ArffData/%s_test_exp_%s_%d_demo_%b.arff", prefix, data, type, fv, k, demo);
//
//		/***Feature selection based on DF.****/
//		int maxDF = -1, minDF = 0;
//		System.out.println(String.format("Start generating %s features based on DF....", type));
//		UserAnalyzer fs_analyzer = new UserAnalyzer(tokenModel, classNumber, null, Ngram, lengthThreshold, false);
//		fs_analyzer.LoadStopwords(stopwords);
//		fs_analyzer.loadUserDir(tweetTrain, suffix);
//		fs_analyzer.featureSelection(features, "DF", maxDF, minDF, k);
		
		/***Generate training Arff files based on the selected features.***/
		System.out.println(String.format("Start generating %s training tweets....", type));
		SplitUserAnalyzer train_analyzer = new SplitUserAnalyzer(tokenModel, classNumber, features, Ngram, lengthThreshold, false);
		train_analyzer.LoadStopwords(stopwords);
		train_analyzer.loadUserDir(tweetTrain, suffix);
		train_analyzer.loadIAT(trainIAT);
		train_analyzer.setFeatureValues("TFIDF", 2);
		train_analyzer.generateArffData(trainFile, att, demo);
//		train_analyzer.generateArffData(trainExpFile, "exp", demo);
		
//		/***Generate testing Arff files based on the selected features.***/
//		System.out.println(String.format("Start generating %s testing tweets....", type));
//		UserAnalyzer test_analyzer = new UserAnalyzer(tokenModel, classNumber, features, Ngram, lengthThreshold, false);
//		test_analyzer.loadUserDir(tweetTest, suffix);
//		test_analyzer.loadIAT(testIAT);
//		test_analyzer.setFeatureValues("TFIDF", 2);
//		test_analyzer.generateArffData(testImpFile, "imp", demo);
//		test_analyzer.generateArffData(testExpFile, "exp", demo);
//		
//		ArrayList<String> topFvs = new ArrayList<String>();
//		double[] weights;
//		MyPriorityQueue<_RankItem> rankq = new MyPriorityQueue<_RankItem>(k);
		
		/***Train linear regression models.***/
		try{
			/***Implicit attitudes.****/
			LinearRegression lr = new LinearRegression();
			System.out.println(String.format("Start loading %s training data from %s....", type, trainFile));
			BufferedReader trainReader = new BufferedReader(new FileReader(trainFile));
			Instances train = new Instances(trainReader);
			train.setClassIndex(train.numAttributes() - 1);
			System.out.print("Total number of attributes is "+train.numAttributes());
			lr.buildClassifier(train);

			/***Write out the selected features.**/
			double[] weights = lr.coefficients();
//			for(double v: weights)
//				System.out.println(v);
			
			PrintWriter writer = new PrintWriter(new File(String.format("%s/%s/models/%s_weights_df_%d_imp_demo_%b_%d.txt", prefix, data, type, k, demo, i)));
			for(int w=0; w<weights.length; w++){
				if(w == weights.length - 1)
					writer.write(String.format("BIAS,%.4f\n", weights[w]));
				else
					writer.write(String.format("%s,%.5f\n", train.attribute(w).name(), weights[w]));
			}
			writer.close();
		} catch(Exception e){
			e.printStackTrace();
		}}
	}
}
