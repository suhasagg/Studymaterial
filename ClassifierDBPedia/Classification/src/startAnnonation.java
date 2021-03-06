import java.util.List;

import org.dbpedia.spotlight.model.DBpediaResource;

//Select topics to which entity belongs,entity is recognized using DBpedia spotlight or keyword derivation using openNLP
//After obtaining dbpedia resource, topic corresponding to resource is derived using Sparql Query



public class startAnnonation {
	public static void main(String[] args) throws Exception {
		// String question =
		// "What is the winning chances of BJP in New Delhi elections?";
		String question = "india-call-up-harbhajan-as-cover-for-injured-ashwin";
		db c = new db();
		c.configiration(0.25, 20);
		// , 0, "non", "AtLeastOneNounSelector", "Default", "yes");
		c.evaluate(question);
		System.out.println("resource : " + c.getResu());
		SparqlQueryExecuter e = new SparqlQueryExecuter("http://dbpedia.org",
				"http://dbpedia.org/sparql");
		String example1 = "select ?subject where {dbpedia:Harbhajan_Singh dcterms:subject ?subject}";
		String example2 = "select ?o where ?s ?p ?o {dbpedia:Harbhajan_Singh a dbo:?o}";
		String example3 = "PREFIX dbres: <http://dbpedia.org/resource/>"
				+ "PREFIX rdf:<http://www.w3.org/1/02/22-rdf-syntax-ns#>"
				+ "select ?o where {dbres:Basketball rdf:type ?o} LIMIT 10";

		String example4 = "PREFIX  dbres: <http://dbpedia.org/resource/>"
				+ "\n"
				+ "PREFIX  rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>"
				+ "\n" + "\n" + "SELECT ?o" + "\n" + "WHERE" + "\n" + "\t"
				+ "{ dbres:Basketball rdf:type ?o}" + "\n" + "LIMIT 10";

		String example5 = "PREFIX  dbres: <http://dbpedia.org/resource/>"
				+ "\n"
				+ "PREFIX  rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>"
				+ "\n" + "\n" + "SELECT ?o" + "\n" + "WHERE" + "\n" + "\t"
				+ "{ dbres:Shahrukh_Khan rdf:type ?o}" + "\n" + "LIMIT 10";

		String example6 = "PREFIX  dbres: <http://dbpedia.org/resource/>"
				+ "\n"
				+ "PREFIX category: <http://dbpedia.org/resource/Category:>"
				+ "\n" + "\n" + "SELECT ?o" + "\n" + "WHERE" + "\n" + "\t"
				+ "{ dbres:Shahrukh_Khan category:type ?o}" + "\n" + "LIMIT 10";
		
		 

		String example7 = "PREFIX dbpedia: <http://dbpedia.org/resource/>"
				+ "\n" + "SELECT * WHERE {" + "\n"
				+ "dbpedia:Mortal_Kombat a ?c1 ; a ?c2 ."
				+ "?c1 rdfs:subClassOf ?c2 ." + "}";

		String example8 = "PREFIX  dbres: <http://dbpedia.org/resource/>"
				+ "\n"
				+ "PREFIX  skos: <http://www.w3.org/2004/02/skos/core#>"
				+ "\n"
				+ "PREFIX Category: <http://dbpedia.org/resource/Category:>"
				+ "\n"
				+ "\n"
				+ "SELECT ?o"
				+ "\n"
				+ "WHERE"
				+ "\n"
				+ "\t"
				+ "{ <http://dbpedia.org/page/Category:American_venture_capital_firms> skos:broader ?o}";

		String example9 = "PREFIX  dbres: <http://dbpedia.org/resource/>"
				+ "\n"
				+ "PREFIX category: <http://dbpedia.org/resource/Category:>"
				+ "\n" + "\n" + "SELECT ?o" + "\n" + "WHERE" + "\n" + "\t"
				+ "{ dbres:Shahrukh_Khan category:type ?o}" + "\n" + "LIMIT 10";

		String example10 = "PREFIX dbpedia: <http://dbpedia.org/resource/>"
				+ "\n" + "SELECT ?c1 WHERE {" + "\n"
				+ "dbpedia:Shahrukh_Khan a ?c1 ; a ?c2 ."
				+ "?c1 rdfs:subClassOf ?c2 ." + "}";

		String example11 = "PREFIX  dbres: <http://dbpedia.org/resource/>"
				+ "\n" + "PREFIX dcterms: <http://purl.org/dc/terms/>" + "\n"
				+ "PREFIX category: <http://dbpedia.org/resource/Category:>"
				+ "\n" + "\n" + "SELECT ?o" + "\n" + "WHERE" + "\n" + "\t"
				+ "{?o dcterms:subject dbres:Shahrukh_Khan}";

		String example12 = "PREFIX  dbres: <http://dbpedia.org/resource/>"
				+ "\n"
				+ "PREFIX  rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>"
				+ "\n" + "\n" + "SELECT ?o" + "\n" + "WHERE" + "\n" + "\t"
				+ "{ dbres:Mortal_Kombat rdf:type ?o}" + "\n" + "LIMIT 10";

		String example17 = "PREFIX  dbres: <http://dbpedia.org/resource/>"
				+ "\n"
				+ "PREFIX  rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>"
				+ "\n" + "\n" + "SELECT ?o" + "\n" + "WHERE" + "\n" + "\t"
				+ "{ dbres:Srinagar rdf:type ?o}" + "\n" + "LIMIT 10";

		

		String question1 = "Ladakh Amazing Tour Package. Trip package to Srinagar,Sonamarg,Zozilla Pass,Nubra for 10 Days/ 9 Nights from New Delhi, Indi";
		String question2 = "Book Rishikesh Tour Packages, Rishikesh Holiday Tour Packages - HelloTravel";

		db c1 = new db();
		// c1.configiration(0.25,20);
		c1.configiration(0.0, 0);
		// , 0, "non", "AtLeastOneNounSelector", "Default", "yes");
		c1.evaluate(question2);
		System.out.println("resource : " + c1.getResu());

		List<DBpediaResource> uris = e.query(example17);
		System.out.println(uris);

	}
}