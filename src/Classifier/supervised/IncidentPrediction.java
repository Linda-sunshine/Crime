package Classifier.supervised;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;

import weka.core.Instance;

public class IncidentPrediction {
	double[] m_weights;
	
	// load the trained weights from the files.
	public void loadWeights(String filename){
		if (filename==null || filename.isEmpty())
			return;
		
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String line;
			ArrayList<Double> ws = new ArrayList<Double>();
			while ((line = reader.readLine()) != null) {
				ws.add(Double.valueOf(line));
			}
			reader.close();
			m_weights = new double[ws.size()];
			for(int i=0; i<ws.size(); i++)
				m_weights[i] = ws.get(i);
			System.out.format("%d feature weights loaded from %s...\n", m_weights.length, filename);
		} catch (IOException e) {
			System.err.format("[Error]Failed to open file %s!!", filename);
		}
	}
	
	// The prediction function with given model weights. 
	public double classify(Instance ins){
		if(m_weights.length == 0 || m_weights == null){
			System.err.println("Load model weights first!");
			return -1;
		}
		double pred = 0;
		// dot product
		for(int i=0; i<=ins.numAttributes(); i++){
			// bias term
			if(i == ins.numAttributes()){
				pred += m_weights[i];
				return pred;
			} else{
				if(ins.value(i) != 0){
					pred += m_weights[i]*ins.value(i);
				}
			}
		}
		return pred;
	}
	
	
}
