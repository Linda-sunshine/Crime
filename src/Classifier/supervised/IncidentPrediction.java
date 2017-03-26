package Classifier.supervised;
import weka.core.Instance;

public class IncidentPrediction {
	double[] m_weights;
	
	// load the trained weights from the files.
	public void loadWeights(String filename){
		
	}
	
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
