package mains;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import Analyzer.UserAnalyzer;
import structures.MyPriorityQueue;
import structures._RankItem;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.core.Instance;
import weka.core.Instances;
import org.apache.commons.math3.*;
import org.apache.commons.math3.stat.inference.TTest;


public class JWeightMain {
	public static void main(String[] args) throws Exception{

		String tokenModel = "./data/Model/en-token.bin"; // Token model.
		String stopwords = "./data/Model/stopwords.dat";
		
		int k = 2000;
//		String prefix = "/if15/lg5bt/DSIData";//"./data"
		String prefix = "./data";
		String data = "geo";

		String fv = "toplr";
		for(String type: new String[]{"black", "gay"}){
			for(String att: new String[]{"imp", "exp"}){
		
//		String type = "gay";// "black" or "gay"
		String suffix = ".csv";
//		String att = "imp";
		boolean demo = false;// whether we include the demo in the training.
		
		String trainFile = String.format("%s/%s/ArffData/%s_train_%s_%s_%d_demo_%b.arff", prefix, data, type, att, fv, k, demo);	
		String testFile = String.format("%s/%s/ArffData/%s_test_%s_%s_%d_demo_%b.arff", prefix, data, type, att, fv, k, demo);
		String sortFile = String.format("%s/%s/Results/%s_sort_%s_%s_%d_demo_%b.txt", prefix, data, type, att, fv, k, demo);

		System.out.print(String.format("[Info]k:%d,data:%s,fv:%s,type:%s,demo:%b,att:%s\n",k,data,fv,type,demo,att));
		
//		/***Generate training Arff files based on the selected features.***/
//		System.out.println(String.format("Start generating %s training tweets....", type));
//		UserAnalyzer train_analyzer = new UserAnalyzer(tokenModel, classNumber, features, Ngram, lengthThreshold, false);
//		train_analyzer.LoadStopwords(stopwords);
//		train_analyzer.loadUserDir(tweetTrain, suffix);
//		train_analyzer.loadIAT(trainIAT);
//		train_analyzer.setFeatureValues("TFIDF", 2);
//		train_analyzer.generateArffData(trainFile, att, demo);
//		
//		/***Generate testing Arff files based on the selected features.***/
//		System.out.println(String.format("Start generating %s testing tweets....", type));
//		UserAnalyzer test_analyzer = new UserAnalyzer(tokenModel, classNumber, features, Ngram, lengthThreshold, false);
//		test_analyzer.loadUserDir(tweetTest, suffix);
//		test_analyzer.loadIAT(testIAT);
//		test_analyzer.setFeatureValues("TFIDF", 2);
//		test_analyzer.generateArffData(testFile, att, demo);
		
		LinearRegression lr = new LinearRegression();
		
		System.out.println(String.format("Start loading %s training data from %s....", type, trainFile));
		BufferedReader trainReader = new BufferedReader(new FileReader(trainFile));
		Instances train = new Instances(trainReader);
		train.setClassIndex(train.numAttributes() - 1);
		lr.buildClassifier(train);

		System.out.println(String.format("Start loading %s testing data from %s....", type, testFile));
		BufferedReader testReader = new BufferedReader(new FileReader(testFile));
		Instances test = new Instances(testReader);
		test.setClassIndex(test.numAttributes() - 1);

		System.out.println("Start evaluation...");
		Evaluation eval = new Evaluation(train);
		eval.evaluateModel(lr, test);
		System.out.println(eval.toSummaryString(String.format("\nResults For %s Attitudes\n======\n", att), false));

		// seed words list.
		double[] weights = lr.coefficients();
		MyPriorityQueue<_RankItem> rankq = new MyPriorityQueue<_RankItem>(k);
		for(int i=0; i<weights.length-1; i++){
			if(weights[i] != 0)
				rankq.add(new _RankItem(i, weights[i]));
		}
		double[] zeros = new double[weights.length];
		TTest t = new TTest();
		double pVal = t.pairedTTest(weights, zeros);
		System.out.println(String.format("[Info]pValue is %.5f.", pVal));
		
		try{
			PrintWriter writer = new PrintWriter(new File(sortFile));
			for(_RankItem it: rankq){
				writer.write(String.format("%s, %.4f\n", train.attribute(it.m_index).name(), it.m_value));
			}
			writer.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}
}}}
