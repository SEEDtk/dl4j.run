/**
 *
 */
package org.theseed.dl4j.train;

import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * This computes the prediction error for a classification model.  The error is the number of cases where the
 * best class is incorrectly predicted divided by the total number of cases.
 *
 * @author Bruce Parrello
 *
 */
public class ClassPredictError implements IPredictError {

    // FIELDS
    /** number of incorrect predictions */
    private int errors;
    /** total number of predictions */
    private int rows;
    /** number of labels */
    private int cols;

    public ClassPredictError(List<String> labels) {
        this.errors = 0;
        this.rows = 0;
        this.cols = labels.size();
    }

    @Override
    public void accumulate(INDArray expect, INDArray output) {
        // Loop through the data rows.
        for (int r = 0; r < expect.rows(); r++) {
            int expected = computeBest(expect, r);
            int predicted = computeBest(output, r);
            if (expected != predicted)
                this.errors++;
            this.rows++;
        }
    }

    /**
     * Compute the index of the best label in the specified row.
     *
     * @param data	array containing data rows, one column per label
     * @param r		row to examine
     *
     * @return the index of the highest value in the row
     */
    private int computeBest(INDArray data, int r) {
        double best = data.getDouble(r, 0);
        int retVal = 0;
        for (int i = 1; i < this.cols; i++) {
            double val = data.getDouble(r, i);
            if (val > best) {
                best = val;
                retVal = i;
            }
        }
        return retVal;
    }

    @Override
    public double getError() {
        double retVal = 0.0;
        if (this.errors > 0)
            retVal = ((double) this.errors) / this.rows;
        return retVal;
    }

    @Override
    public void finish() {
    }

}
