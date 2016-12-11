package Regression;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.Writer;
import java.util.ArrayList;

import structures._Review;
import structures._SparseFeature;
import structures._User;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.*;
import weka.core.*;

public class MyLinearRegression {
	static LinearRegression m_lr;
	Instances m_instances;
	Instances m_testInstances;

	int m_featureSize;
	public static void main(String[] args){
		try {
		
			
			
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}		
	}
	
	public MyLinearRegression(ArrayList<_User> users, int featureSize){
		m_lr = new LinearRegression();
		m_featureSize = featureSize;
		init(users);
	}
	
	public void init(ArrayList<_User> users){
		m_instances = new Instances(null, null, users.size());
		for(_User u: users){
			m_instances.add(createOneInstance(u));
		}
	}

	// 0-th feature is the bias, last one is the iat score.
	public Instance createOneInstance(_User u){
		double[] vs = new double[m_featureSize];
		for(_Review r: u.getReviews()){
			for(_SparseFeature sf: r.getSparse()){
				vs[sf.getIndex()] += sf.getValue();
			}
		}
		
		Instance in = new DenseInstance(vs.length+2);
		in.setValue(0, 0);// bias
		for(int i=0; i<vs.length; i++){
			in.setValue(i+1, vs[i]);
		}
		in.setValue(vs.length+1, u.getIATScore());
		return in;
	}
	
	// 0-th feature is the bias, last one is the iat score.
	public Instance createOneTestInstance(_User u){
		double[] vs = new double[m_featureSize];
		for(_Review r: u.getReviews()){
			for(_SparseFeature sf: r.getSparse()){
				vs[sf.getIndex()] += sf.getValue();
			}
		}
		Instance in = new DenseInstance(vs.length+1);
		in.setValue(0, 0);// bias
		for(int i=0; i<vs.length; i++){
			in.setValue(i+1, vs[i]);
		}
		return in;
	}
	public void train(){
		try {
			m_lr.buildClassifier(m_instances);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public void test(ArrayList<_User> users){
		m_testInstances = new Instances(null, null, users.size());
		for(_User u: users){
			Instance in = createOneTestInstance(u);
			try {
				double val = m_lr.classifyInstance(in);
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			System.out.println();
		}
	}
}
