/**
 * 
 */
package org.theseed.dl4j;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.commons.lang3.StringUtils;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.kohsuke.args4j.Argument;
import org.kohsuke.args4j.Option;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.theseed.basic.ParseFailureException;
import org.theseed.io.LineReader;
import org.theseed.io.TabbedLineReader;
import org.theseed.utils.BaseReportProcessor;
import org.theseed.utils.Parms;

/**
 * This report determines the mean bias for each input column of a classification processor.  For
 * each output label, the mean value of each input column is computed.  By comparing the means, we
 * can determine the bias of each input column toward a given output label.
 * 
 * This performs for a classification processor a function similar to that performed by the
 * PearsonProcessor for regression.  Note that we do not create a standard ModelProcessor since
 * we don't know if it's random-forest or neural-net.  Instead, we use the known fixed file names
 * in the directory. 
 * 
 * The positional parameter is the name of the input model directory.  The command-line options are
 * as follows.
 * 
 * -h	display command-line usage
 * -v	display more frequent log messages
 * -o	output file for report (if not STDOUT)
 * -m	include metadata columns from the specified file; the first column is presumed to be a column name 
 * 		from the model
 * 
 * @author Bruce Parrello
 *
 */
public class MeanBiasProcessor extends BaseReportProcessor {
	
	// FIELDS
	/** logging facility */
	protected static Logger log = LoggerFactory.getLogger(MeanBiasProcessor.class);
	/** metadata column names */
	private String metaHeaders;
	/** default metadata value */
	private String metaDefault;
	/** map of column names to metadata values */ 
	private Map<String, String> metaValues;
	/** array of labels */
	private String[] labels;
	/** array of column names */
	private String[] colNames;
	/** set of metadata column names in the training file */
	private Set<String> metaCols;
	/** map of column names to label sums */
	private SortedMap<String, SummaryStatistics[]> meanMap;
	/** training file */
	private File trainFile;
	/** label column name */
	private String labelCol;
	
	// COMMAND-LINE OPTIONS
	
	/** optional metadata input file */
	@Option(name = "--metadata", aliases = { "-m", "--meta" }, metaVar = "impact.tbl", usage = "metadata file for column names")
	private File metaFile;
	
	/** model directory name */
	@Argument(index = 0, metaVar = "modelDir", usage = "model input directory")
	private File modelDir;
	
	/**
	 * Run a mean-bias analysis for a specified training file in a specified model directory.
	 * 
	 * @param modelDir		input model directory
	 * @param trainFile		training file to use
	 * 
	 * @throws ParseFailureException
	 * @throws IOException 
	 */
	public void analyzeLabelBias(File modelDir, File trainFile) throws IOException, ParseFailureException {
		this.trainFile = trainFile;
		this.modelDir = modelDir;
		// Denote we have no metadata file.
		this.clearMetaFileData();
		// Read in the model directory specifications.
		this.processModelDir();
		// Fill in the summary statistics for each label.
		this.computeBias();
	}
	
	/**
	 * @return the ordered list of labels.
	 */
	public String[] getLabels() {
		return this.labels;
	}

	/**
	 * @return the mean for each label in each column
	 */
	public Map<String, double[]> getMeans() {
		Map<String, double[]> retVal = new TreeMap<String, double[]>();
		for (Map.Entry<String, SummaryStatistics[]> meanEntry : this.meanMap.entrySet()) {
			String colName = meanEntry.getKey();
			double[] means = Arrays.stream(meanEntry.getValue()).mapToDouble(x -> x.getMean())
					.toArray();
			retVal.put(colName, means);
		}
		return retVal;
	}
	
	@Override
	protected void setReporterDefaults() {
		this.metaFile = null;
	}

	@Override
	protected void validateReporterParms() throws IOException, ParseFailureException {
		// Validate the metadata file.
		if (this.metaFile == null) {
			clearMetaFileData();
		} else if (! this.metaFile.canRead())
			throw new FileNotFoundException("Metadata file " + this.metaFile + " is not found or unreadable.");
		else {
			log.info("Reading metadata from {}.", this.metaFile);
			try (TabbedLineReader metaStream = new TabbedLineReader(this.metaFile)) {
				// Start with the headers.
				String[] headers = metaStream.getLabels();
				if (headers.length < 2)
					throw new IOException("Meta-data file must contain at least two columns.");
				// Everything but the first column is a metadata header.  Note that there is a leading tab for
				// a metadata string, because it is inserted in the middle of the output.
				int n = headers.length;
				this.metaHeaders = "\t" + IntStream.range(1, n).mapToObj(i -> headers[i]).collect(Collectors.joining("\t"));
				// The default is empty strings in every column.
				this.metaDefault = StringUtils.repeat("\t", n - 1);
				this.metaValues = new HashMap<String, String>(50);
				for (TabbedLineReader.Line line : metaStream) {
					String[] fields = line.getFields();
					// Pull out the column ID, then remove it from the output string.
					String columnId = fields[0];
					fields[0] = "";
					// The output string is all the fields joined together.
					String output = StringUtils.join(fields, "\t");
					this.metaValues.put(columnId, output);
				}
				log.info("{} metadata strings read from {}.", this.metaValues.size(), this.metaFile);
			}
		}
		// Get the training file.
		this.trainFile = new File(this.modelDir, "training.tbl");
		// Get all the data from the model directory.
		this.processModelDir();
	}

	/**
	 * Denote we have no metadata file.
	 */
	protected void clearMetaFileData() {
		this.metaHeaders = "";
		this.metaDefault = "";
		this.metaValues = Collections.emptyMap();
	}

	/**
	 * Extract all the data we need from the model directory.
	 * 
	 * @throws IOException
	 * @throws ParseFailureException
	 */
	protected void processModelDir() throws FileNotFoundException, IOException, ParseFailureException {
		// Check the model directory.
		File labelFile = new File(this.modelDir, "labels.txt");
		File parmFile = new File(this.modelDir, "parms.prm");
		if (! labelFile.exists() || ! this.trainFile.exists() || ! parmFile.exists())
			throw new FileNotFoundException(this.modelDir + " does not look like a model directory.");
		if (! this.trainFile.canRead())
			throw new FileNotFoundException("Training file in " + this.modelDir + " is unreadable.");
		// Get the label list.
		List<String> labels = LineReader.readList(labelFile);
		this.labels = new String[labels.size()];
		this.labels = labels.toArray(this.labels);
		// Compute the metadata column names in the training file.  These will be ignored during
		// processing.  Note that the label column is treated as a metadata column as well, since
		// we do not compute its mean.  We do, however, need to remember it for other reasons.
		Parms parms = new Parms(parmFile);
		String metaString = parms.getValue("--meta");
		this.metaCols = Arrays.stream(StringUtils.split(metaString, ",")).collect(Collectors.toSet());
		this.labelCol = parms.getValue("--col");
		if (this.labelCol.isEmpty())
			throw new ParseFailureException("Missing label column in parm file " + parmFile + ".");
		this.metaCols.add(this.labelCol);
		log.info("{} metadata columns present in training file.", this.metaCols.size());
	}

	@Override
	protected void runReporter(PrintWriter writer) throws Exception {
		// Determine the bias for each input column.
		this.computeBias();
		// Now we produce the output.
		writeReport(writer);
	}

	/**
	 * Write the output report.
	 * 
	 * @param writer	PrintWriter to receive the output
	 */
	public void writeReport(PrintWriter writer) {
		log.info("Writing output.");
		// Form the headers.
		writer.format("column_name%s\t%s%n", this.metaHeaders, StringUtils.join(this.labels, "\t"));
		// Output each column.
		for (Map.Entry<String, SummaryStatistics[]> entry : this.meanMap.entrySet()) {
			String colName = entry.getKey();
			String metaData = this.metaValues.getOrDefault(colName, metaDefault);
			String stats = Arrays.stream(entry.getValue()).map(x -> String.format("%6.4f", x.getMean()))
					.collect(Collectors.joining("\t"));
			writer.format("%s%s\t%s%n", colName, metaData, stats);
		}
	}

	/**
	 * Compute the bias for each input column and build the summary statistics.
	 * 
	 * @throws IOException
	 */
	protected void computeBias() throws IOException {
		// Start reading the training file.
		log.info("Reading {}.", this.trainFile);
		try (TabbedLineReader trainStream = new TabbedLineReader(this.trainFile)) {
			// Find the label column.
			int labelColIdx = trainStream.findField(this.labelCol);
			// Get the array of column names.
			this.colNames = trainStream.getLabels();
			// Initialize the output map.
			this.meanMap = new TreeMap<String, SummaryStatistics[]>();
			for (String colName : this.colNames) {
				if (! this.metaCols.contains(colName)) {
					// Here we have an input column.  Create the summary stats.
					SummaryStatistics[] summStats = IntStream.range(0, this.labels.length)
							.mapToObj(i -> new SummaryStatistics()).toArray(SummaryStatistics[]::new);
					this.meanMap.put(colName, summStats);
				}
			}
			// This will count the number of input lines.
			int rowCount = 0;
			// This will count the number of input values.
			int valCount = 0;
			// Loop through the file, processing columns.
			for (TabbedLineReader.Line line : trainStream) {
				// Find this line's label.
				String label = line.get(labelColIdx);
				int labelIdx = this.labels.length - 1;
				while (labelIdx >= 0 && ! this.labels[labelIdx].contentEquals(label)) labelIdx--;
				if (labelIdx < 0)
					throw new IOException("Invalid label value \"" + label + "\" encountered in input.");
				rowCount++;
				// Process each column individually.
				for (int i = 0; i < this.colNames.length; i++) {
					String colName = this.colNames[i];
					// Get the summary statistics for this column.
					SummaryStatistics[] summStats = this.meanMap.get(colName);
					if (summStats != null) {
						// Here we have a real input column.
						double val = line.getDouble(i);
						summStats[labelIdx].addValue(val);
						valCount++;
					}
				}
			}
			log.info("{} rows read, {} values processed.", rowCount, valCount);
		}
	}

}
