package Application;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;

public class DataAnalysis {
	ArrayList<Pair> m_stat;
	class Pair{
		protected String county;
		public int num;
		
		public Pair(String c, int n){
			county = c;
			num = n;
		}
	}

	public void loadFile(String filename){
		try {
			File file = new File(filename);
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
			String line;			
			String[] strs;

			// Skip the first line since it is user name.
			reader.readLine(); 
			m_stat = new ArrayList<Pair>();
			while((line = reader.readLine()) != null){
				if(line.endsWith("---")){
					strs = line.split(" ");
					if(strs.length >1)
						m_stat.add(new Pair(extractID(strs[0]), Integer.parseInt(strs[3])));
				}
			}
			reader.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	
	public void analyze(){
		
		Collections.sort(m_stat, new Comparator<Pair>(){
			@Override
			public int compare(Pair p1, Pair p2){
				return p2.num - p1.num;
			}
		});
		System.out.println("Max: "+ m_stat.get(0).num + "\t" + m_stat.get(0).county);
//		for(Pair p: m_stat)
//			System.out.print(p.num+"\t");
		int[] counts = new int[20];
		for(Pair p: m_stat){
			counts[p.num/10]++;
		}
		for(int c: counts)
			System.out.print(c+"\t");
	}
	
	public String extractID(String name){
		int index = 0;
		while(!Character.isAlphabetic(name.charAt(index))){
			index++;
			if(index >= name.length())
				System.out.println("bug");
		}
		return name.substring(index);
	}

}
