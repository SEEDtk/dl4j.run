/**
 *
 */
package org.theseed.dl4j;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.factory.Nd4j;
import org.theseed.io.TabbedLineReader;

/**
 * This class reads a training set in batches.  The training set is a tab-delimited input file
 * with headers.  Each record contains represents an example.  An example consists of one or
 * more numeric sensor values called a feature and a string called a label.  The constructor
 * specifies the list of permissible label values and the name or index of the column in which
 * the label is placed.  Usually this is 0, indicating the last column.
 *
 * The primary processing function is to output a dataset representing a batch of input.  This
 * dataset can then be used to train or test a model.  The default batch size is 100.  This can
 * be modified by the client.  In addition
 *
 * @author Bruce Parrello
 *
 */
public class TabbedTrainingSetReader implements Iterable<DataSet>, Iterator<DataSet> {

    // FIELDS
    /** input tabbed file */
    TabbedLineReader reader;
    /** list of valid labels */
    ArrayList<String> labels;
    /** column index of label column */
    int labelIdx;
    /** normalizer to be applied to all batches of input */
    DataNormalization normalizer;
    /** current batch size */
    int batchSize;
    /** buffer array for holding input */
    ArrayList<Entry> buffer;

    /** This is a simple class for holding a feature and its label. */
    private class Entry {
        double[] feature;
        int label;

        /** Create a blank entry. */
        public Entry() {
            this.feature = new double[reader.size() - 1];
            this.label = 0;
        }
    }

    /**
     * Construct a training set reader for a file.
     *
     * @param file		the file containing the training set
     * @param labelCol	the name or index of the column containing the label
     * @param labels	a list of the acceptable labels, in order
     *
     * @throws IOException
     */
    public TabbedTrainingSetReader(File file, String labelCol, List<String> labels) throws IOException {
        this.reader = new TabbedLineReader(file);
        this.setup(labelCol, labels);
    }

    /**
     * Construct a training set reader for an input stream.
     *
     * @param stream	the stream containing the training set
     * @param labelCol	the name or index of the column containing the label
     * @param labels	a list of the acceptable labels, in order
     *
     * @throws IOException
     */
    public TabbedTrainingSetReader(InputStream stream, String labelCol, List<String> labels) throws IOException {
        this.reader = new TabbedLineReader(stream);
        this.setup(labelCol, labels);
    }

    /**
     * Initialize the fields of this object.
     *
     * @param labelCol	the name or index of the column containing the label
     * @param labels	a list of the acceptable labels, in order
     *
     * @throws IOException
     */
    private void setup(String labelCol, List<String> labels) throws IOException {
        this.labels = new ArrayList<String>(labels);
        this.labelIdx = reader.findField(labelCol);
        this.normalizer = null;
        this.setBatchSize(100);
    }

    @Override
    public Iterator<DataSet> iterator() {
        return this;
    }

    /**
     * Return TRUE if there is more data in this file, else FALSE.
     */
    @Override
    public boolean hasNext() {
        return this.reader.hasNext();
    }

    /**
     * Return the next batch of data.
     */
    @Override
    public DataSet next() {
        // Get the number of fields in each record.
        int n = this.reader.size();
        // The array list should be empty.  Fill it from the input.
        for (int numRead = 0; numRead < this.batchSize && this.hasNext(); numRead++) {
            TabbedLineReader.Line line = this.reader.next();
            Entry record = new Entry();
            int pos = 0;
            for (int i = 0; i < n; i++) {
                if (i != this.labelIdx) {
                    // Here we have a feature column.
                    record.feature[pos++] = line.getDouble(i);
                } else {
                    // We have a label.  Translate from a string to a number.
                    String labelName = line.get(i);
                    int label = this.labels.indexOf(labelName);
                    if (label < 0) {
                        throw new IllegalArgumentException("Invalid label " + labelName);
                    } else {
                        record.label = label;
                    }
                }
            }
            this.buffer.add(record);
        }
        // Create and fill the feature and label arrays.
        NDArray features = new NDArray(this.buffer.size(), n - 1);
        INDArray labels = Nd4j.zeros(this.buffer.size(), this.labels.size());
        int row = 0;
        for (Entry record : this.buffer) {
            features.putRow(row, Nd4j.create(record.feature));
            labels.put(row, record.label, 1.0);
            row++;
        }
        this.buffer.clear();
        // Build the dataset.
        DataSet retVal = new DataSet(features, labels);
        if (this.normalizer != null)
            this.normalizer.transform(retVal);
        return retVal;
    }

    /**
     * @return the normalizer
     */
    public DataNormalization getNormalizer() {
        return this.normalizer;
    }

    /**
     * @return the batch size
     */
    public int getBatchSize() {
        return this.batchSize;
    }

    /**
     * @param normalizer 	normalizer to apply to each incoming set
     */
    public TabbedTrainingSetReader setNormalizer(DataNormalization normalizer) {
        this.normalizer = normalizer;
        return this;
    }

    /**
     * @param batchSize 	new batch size for reading
     */
    public TabbedTrainingSetReader setBatchSize(int batchSize) {
        this.batchSize = batchSize;
        this.buffer = new ArrayList<Entry>(batchSize);
        return this;
    }

    /**
     * @return the number of sensor values per feature
     */
    public int width() {
        return this.reader.size() - 1;
    }

}
