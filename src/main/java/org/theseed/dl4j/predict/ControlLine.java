/**
 *
 */
package org.theseed.dl4j.predict;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import org.apache.commons.lang3.StringUtils;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.theseed.dl4j.ChannelDataSetReader;
import org.theseed.dl4j.TabbedDataSetReader;
import org.theseed.io.TabbedLineReader;

/**
 * This class represents a single request to process a model against training data.
 * It contains the model and input file names, the desired output column name,
 * the output column index, and the list of metadata columns.
 *
 * @author Bruce Parrello
 *
 */
public class ControlLine {

    // FIELDS

    /** model file name */
    private File modelFile;
    /** input file name */
    private File inputFile;
    /** metadata column names; the first is the key column */
    private List<String> metaCols;
    /** output column name */
    private String outputHeading;


    /** model file name column index */
    private static int MODEL_FILE_COL = 0;
    /** input file name column index */
    private static int INPUT_FILE_COL = 1;
    /** metadata column name list column index */
    private static int METADATA_COL = 2;
    /** output column name column index */
    private static int OUTPUT_NAME_COL = 3;
    /** number of columns */
    private static int COL_COUNT = 4;
    /** neural net model */
    private MultiLayerNetwork model;
    /** input reader */
    private TabbedDataSetReader reader;

    /**
     * @return the number of input columns
     */
    public static int colCount() {
        return COL_COUNT;
    }

    /**
     * Construct a control object from an input line.
     *
     * @param line		input line
     * @param modelDir	target model directory
     * @param keyName	key column name
     */
    public ControlLine(TabbedLineReader.Line line, File modelDir, String keyName) {
        this.modelFile = new File(modelDir, line.get(MODEL_FILE_COL));
        this.inputFile = new File(modelDir, line.get(INPUT_FILE_COL));
        this.outputHeading = line.get(OUTPUT_NAME_COL);
        // Build the metadata column list.  The first column is the key column.
        String [] otherMeta = StringUtils.split(line.get(METADATA_COL), ',');
        this.metaCols = new ArrayList<String>(otherMeta.length + 1);
        this.metaCols.add(keyName);
        this.metaCols.addAll(Arrays.asList(otherMeta));
        // Clear the file handlers.
        this.model = null;
        this.reader = null;
    }

    /**
     * Initialize the model and the reader for processing.
     *
     * @param channelMap	channel map for input, or NULL if the input is not channeled
     *
     * @throws IOException
     */
    public void init(Map<String, double[]> channelMap) throws IOException {
        // Read in the model and the normalizer.
        this.model = ModelSerializer.restoreMultiLayerNetwork(this.modelFile, false);
        DataNormalization normalizer = ModelSerializer.restoreNormalizerFromFile(this.modelFile);
        // Open the input reader.
        if (channelMap == null) {
            this.reader = new TabbedDataSetReader(this.inputFile, this.metaCols);
        } else {
            this.reader = new ChannelDataSetReader(this.inputFile, this.metaCols, channelMap);
        }
        // Attach the normalizer.
        this.reader.setNormalizer(normalizer);
    }

    /**
     * Release the memory for the model and the reader.
     */
    public void close() {
        this.reader = null;
        this.model = null;
    }

    /**
     * @return the model
     */
    public MultiLayerNetwork getModel() {
        return model;
    }

    /**
     * @return the dataset reader
     */
    public TabbedDataSetReader getReader() {
        return reader;
    }

    /**
     * @return the outputHeading
     */
    public String getOutputHeading() {
        return outputHeading;
    }

}
