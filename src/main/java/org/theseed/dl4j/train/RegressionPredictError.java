/**
 *
 */
package org.theseed.dl4j.train;

import java.util.Arrays;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.theseed.counters.RegressionStatistics;

/**
 * This class computes the prediction error for a regression model.  For each output, the mean absolute
 * error is computed as a fraction of the total value range.  The prediction error is then the average
 * of these error values for all of the outputs.
 *
 * @author Bruce Parrello
 *
 */
public class RegressionPredictError implements IPredictError {

    // FIELDS
    /** accumulated sums of the prediction errors for each output */
    private double[] sums;
    /** minimum value for each output */
    private double[] mins;
    /** maximum value for each output */
    private double[] maxs;
    /** number of rows counted */
    private int rows;
    /** number of label columns */
    private int cols;
    /** stats producer */
    private RegressionStatistics[] statsTracker;

    /**
     * Construct a new prediction-error object for the specified output labels.
     *
     * @param labels	list of output labels
     */
    public RegressionPredictError(List<String> labels, int rows) {
        this.cols = labels.size();
        this.sums = new double[this.cols];
        Arrays.fill(this.sums, 0.0);
        this.mins = new double[this.cols];
        Arrays.fill(this.mins, Double.MAX_VALUE);
        this.maxs = new double[this.cols];
        Arrays.fill(this.maxs, -Double.MAX_VALUE);
        this.rows = rows;
        this.statsTracker = new RegressionStatistics[this.cols];
        for (int i = 0; i < this.cols; i++)
            this.statsTracker[i] = new RegressionStatistics(rows);
    }

    @Override
    public void accumulate(INDArray expect, INDArray output) {
        // Loop through the labels.
        for (int i = 0; i < this.cols; i++) {
            // Loop through the data;
            for (int r = 0; r < expect.rows(); r++) {
                double e = expect.getDouble(r, i);
                double o = output.getDouble(r, i);
                this.sums[i] += Math.abs(e - o);
                // We adjust the max and min based on the expected value.
                if (e > this.maxs[i]) this.maxs[i] = e;
                if (e < this.mins[i]) this.mins[i] = e;
                // Save for the regression statistics.
                this.statsTracker[i].add(e, o);
            }
        }
    }

    @Override
    public double getError() {
        double totalError = 0.0;
        for (int i = 0; i < this.cols; i++) {
            // Compute the relative error for this label.
            double pctError = this.sums[i] / (this.rows * this.getScale(i));
            totalError += pctError;
        }
        // Return the mean relative error.
        return totalError / this.cols;
    }

    @Override
    public void finish() {
        for (int i = 0; i < this.cols; i++) {
            double scaleFactor = this.getScale(i);
            this.statsTracker[i].scale(scaleFactor);
            this.statsTracker[i].finish();
        }
    }

    /**
     * @return the scale factor to divide into the errors in the specified column
     *
     * @param i		index of the column to examine
     */
    private double getScale(int i) {
        double retVal = 1.0;
        if (this.mins[i] < this.maxs[i]) {
            // Normal case.  There is a non-zero value range.
            retVal = (this.maxs[i] - this.mins[i]);
        } else if (this.mins[i] == this.maxs[i]) {
            // Here all the expected values are the same.
            if (this.mins[i] == 0.0) {
                // The expected value is 0, so the error value is not scaled.
                retVal = 1.0;
            } else {
                // Here we scale by the absolute expected value.
                retVal = Math.abs(this.mins[i]);
            }
        }
        return retVal;
    }

    @Override
    public String[] getTitles() {
        return new String[] { "trimean", "trimmedMean", "IQR" };
    }

    @Override
    public double[] getStats() {
        double triMean = 0.0;
        double trimmedMean = 0.0;
        double iqr = 0.0;
        for (int i = 0; i < this.cols; i++) {
            triMean += this.statsTracker[i].trimean();
            trimmedMean += this.statsTracker[i].trimmedMean(0.2);
            iqr += this.statsTracker[i].iqr();
        }
        return new double[] { triMean / this.cols, trimmedMean / this.cols, iqr / this.cols };
    }

}
