/**
 *
 */
package org.theseed.dl4j;

import java.util.Set;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.theseed.counters.CountMap;
import org.theseed.io.TabbedLineReader;

/**
 * This object tracks accuracy information for a model feature.  For each feature, it contains the column name,
 * a flag indicating if the column is a valid on/off column, a count of the ON rows found, and the total
 * absolute error for the ON rows found.  It also contains counts for the training/testing set, indicating how
 * often the feature occurs with each other feature.  These counts are used to compute statistics about the
 * breadth of data for the feature.
 *
 * @author Bruce Parrello
 *
 */
public class AccuracyItem implements Comparable<AccuracyItem> {

    // FIELDS
    /** logging facility */
    protected static Logger log = LoggerFactory.getLogger(AccuracyItem.class);
    /** name of column */
    private String colName;
    /** index of column (0-based) */
    private int colIdx;
    /** TRUE if the column is a valid on/off column */
    private boolean valid;
    /** map containing rows found for each feature in rows containing this feature */
    private CountMap<String> rowCounts;
    /** total absolute error */
    private double totError;
    /** number of predictions found */
    private int predCount;
    /** number of true positive predictions */
    private int tpPredictions;
    /** number of true negative predictions */
    private int tnPredictions;
    /** number of false-positive predictions */
    private int fpPredictions;
    /** number of false-negative predictions */
    private int fnPredictions;
    /** cutoff for high/low classification used to compute pseudo-accuracy */
    protected static double CUTOFF = 1.2;

    /**
     * Construct an accuracy item for a column.  Note we allow construction of an invalid column
     * to force it to be skipped regardless of content.
     *
     * @param name			column name
     * @param idx			column index
     * @param validFlag		TRUE if the column can be valid, else FALSE
     */
    public AccuracyItem(String name, int idx, boolean validFlag) {
        this.colName = name;
        this.valid = validFlag;
        this.colIdx = idx;
        this.rowCounts = new CountMap<String>();
        this.totError = 0.0;
        this.predCount = 0;
        this.tpPredictions = 0;
        this.tnPredictions = 0;
        this.fnPredictions = 0;
        this.fpPredictions = 0;
    }

    /**
     * Record a row for this column.
     *
     * @param active		set of features active in this row
     */
    public void recordRow(Set<String> active) {
        active.stream().forEach(x -> this.rowCounts.count(x));
    }

    /**
     * Return TRUE if this column is on in the specified row.
     *
     * @param line			input line containing the row
     *
     * @return TRUE if the column is on, else FALSE
     */
    public boolean checkRow(TabbedLineReader.Line line) {
        double colVal = line.getDoubleSafe(this.colIdx);
        return (colVal == 1.0);
    }

    /**
     * Record a prediction error for this column.
     *
     * @param actual			actual value
     * @param predicted			predicted value
     */
    public void recordError(double actual, double predicted) {
        this.predCount++;
        this.totError += Math.abs(actual - predicted);
        if (actual >= CUTOFF) {
            if (predicted >= CUTOFF)
                this.tpPredictions++;
            else
                this.fnPredictions++;
        } else {
            if (predicted >= CUTOFF)
                this.fpPredictions++;
            else
                this.tnPredictions++;
        }
    }

    /**
     * @return the accuracy of this column
     */
    public double getMAE() {
        double retVal;
        if (this.predCount == 0)
            retVal = Double.NaN;
        else
            retVal = this.totError / this.predCount;
        return retVal;
    }

    /**
     * @return the column name
     */
    public String getColName() {
        return this.colName;
    }

    /**
     * @return TRUE if this column is valid
     */
    public boolean isValid() {
        return this.valid;
    }

    /**
     * @return the number of rows with the column on
     */
    public int getRowCount() {
        return this.rowCounts.getCount(this.colName);
    }

    /**
     * @return the number of predictions recorded
     */
    public int getPredCount() {
        return this.predCount;
    }

    /**
     * Mark this column as invalid.
     */
    public void invalidate() {
        this.valid = false;
    }

    /**
     * @return the number of other features used with this one
     *
     * @param validSet		set of valid feature names
     */
    public int getWidth(Set<String> validSet) {
        int retVal = (int) validSet.stream().filter(x -> this.rowCounts.getCount(x) > 0).count();
        return retVal;
    }

    /**
     * @return the angle (cosine) between this distribution vector and an even distribution vector
     *
     * @param validSet		set of valid feature names
     */
    public double getBreadth(Set<String> validSet) {
        // Now, get the distance to the mean.
        double tot = 0.0;
        double sqr = 0.0;
        double count = 0.0;
        for (var label : validSet) {
            int n = this.rowCounts.getCount(label);
            if (n > 0 && ! label.equals(this.colName)) {
                sqr += n*n;
                tot += n;
                count++;
            }
        }
        double retVal = tot / (Math.sqrt(sqr) * Math.sqrt(count));
        return retVal;
    }

    @Override
    public int compareTo(AccuracyItem o) {
        int retVal = o.getRowCount() - this.getRowCount();
        if (retVal == 0)
            retVal = this.colName.compareTo(o.colName);
        return retVal;
    }

    /**
     * Specify a new column index for this field.
     *
     * @param colIdx 	the new column index to set
     */
    public void setColIdx(int colIdx) {
        this.colIdx = colIdx;
    }

    /**
     * @return the column index for this field
     */
    public int getColIdx() {
        return this.colIdx;
    }

    /**
     * Set the pseudo-accuracy cutoff.
     *
     * @param cutoff	proposed new cutoff
     */
    public static void setCutoff(double cutoff) {
        CUTOFF = cutoff;
    }

    /**
     * @return the pseudo-accuracy for this field
     */
    public double getAccuracy() {
        double retVal = 0.0;
        if (this.predCount > 0)
            retVal = (this.tnPredictions + this.tpPredictions) / (double) this.predCount;
        if (retVal == 0)
            log.info("Accuracy for {} is zero.", this.colName);
        return retVal;
    }

    /**
     * @return the pseudo-precision for this field
     */
    public double getPrecision() {
        double retVal = 0.0;
        double predPositive = this.tpPredictions + this.fpPredictions;
        if (predPositive > 0.0)
            retVal = this.tpPredictions / predPositive;
        return retVal;
    }

    /**
     * @return the pseudo-recall for this field
     */
    public double getRecall() {
        double retVal = 0.0;
        double realPositive = this.tpPredictions + this.fnPredictions;
        if (realPositive > 0.0)
            retVal = this.tpPredictions / realPositive;
        return retVal;
    }

    /**
     * @return the F1 score for this field
     */
    public double getF1() {
        double retVal = 0.0;
        double recall = this.getRecall();
        double precision = this.getPrecision();
        if (recall > 0.0 || precision > 0.0)
            retVal = (precision * recall) / (precision + recall);
        return retVal;
    }

}
