package mains;

import Application.DataAnalysis;

public class MyDataProcessMain {
	public static void  main(String[] args){
		DataAnalysis ds = new DataAnalysis();
		String seed = "./data/stat/1210gay_seed.txt";
		ds.loadFile(seed);
		ds.analyze();
		
	}

}
