/**
 *
 */
package org.theseed.dl4j;

import java.util.Comparator;

/**
 * This object represents a single prediction/actual pairing.  It can be sorted by prediction
 * value or actual value.
 *
 * @author Bruce Parrello
 *
 */
public class RocItem {

    // FIELDS
    /** prediction value for this item */
    private double predicted;
    /** actual value for this item */
    private double actual;

    /**
     * Comparator to sort by highest prediction.
     */
    public static class ByPredicted implements Comparator<RocItem> {

        @Override
        public int compare(RocItem o1, RocItem o2) {
            int retVal = Double.compare(o2.predicted, o1.predicted);
            if (retVal == 0)
                retVal = Double.compare(o2.actual, o1.actual);
            return retVal;
        }

    }

    /**
     * Comparator to sort by highest actual value.
     */
    public static class ByActual implements Comparator<RocItem> {

        @Override
        public int compare(RocItem o1, RocItem o2) {
            int retVal = Double.compare(o2.actual, o1.actual);
            if (retVal == 0)
                retVal = Double.compare(o2.predicted, o1.predicted);
            return retVal;
        }

    }

    /**
     * Create a ROC item.
     *
     * @param predict	predicted value
     * @param expect	actual value
     */
    public RocItem(double predict, double expect) {
        this.predicted = predict;
        this.actual = expect;
    }

    /**
     * @return the predicted value
     */
    public double getPredicted() {
        return this.predicted;
    }

    /**
     * @return the actual value
     */
    public double getActual() {
        return this.actual;
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        long temp;
        temp = Double.doubleToLongBits(this.actual);
        result = prime * result + (int) (temp ^ (temp >>> 32));
        temp = Double.doubleToLongBits(this.predicted);
        result = prime * result + (int) (temp ^ (temp >>> 32));
        return result;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (!(obj instanceof RocItem)) {
            return false;
        }
        RocItem other = (RocItem) obj;
        if (Double.doubleToLongBits(this.actual) != Double.doubleToLongBits(other.actual)) {
            return false;
        }
        if (Double.doubleToLongBits(this.predicted) != Double.doubleToLongBits(other.predicted)) {
            return false;
        }
        return true;
    }

}
