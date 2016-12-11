package mains;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;

import structures._User;

public class DataProcessMain {
	public static void loadFile(String filename){
		try {
			File file = new File(filename);
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
			String line;			
			String[] strs;

			// Skip the first line since it is user name.
			reader.readLine(); 
			int count = 0;
			String[][] stat = new String[200][2];
			while((line = reader.readLine()) != null){
				if(line.endsWith("---")){
					strs = line.split(" ");
					stat[count][0] = extractID(strs[0]);
					stat[count++][1] = strs[3];
				}
			}
			reader.close();
			
			
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	
	public static String extractID(String name){
		int index = 0;
		while(!Character.isAlphabetic(name.charAt(index)))
			index++;
		return name.substring(index);
	}
	
	public void loadIAT(String filename){
		
	}
	public static void  main(String[] args){
		
	}

}
