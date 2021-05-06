/**
 *
 */
package org.theseed.dl4j;

/**
 * This object describes a cross-reference between two input variables.  It contains the names of the two input
 * columns and their values.
 *
 * @author Bruce Parrello
 *
 */
public class CrossReference implements Comparable<CrossReference> {

    // FIELDS
    /** first column name */
    private String col1Name;
    /** first column value */
    private double col1Value;
    /** second column name */
    private String col2Name;
    /** second column value */
    private double col2Value;

    /**
     * Construct a cross-reference.
     *
     * @param col1		name of the first column
     * @param val1		value of the first column
     * @param col2		name of the second column
     * @param val2		value of the second column
     */
    public CrossReference(String col1, double val1, String col2, double val2) {
        this.col1Name = col1;
        this.col1Value = val1;
        this.col2Name = col2;
        this.col2Value = val2;
    }

    @Override
    public int compareTo(CrossReference o) {
        int retVal = this.col1Name.compareTo(o.col1Name);
        if (retVal == 0) {
            retVal = Double.compare(this.col1Value, o.col1Value);
            if (retVal == 0) {
                retVal = this.col2Name.compareTo(o.col2Name);
                if (retVal == 0)
                    retVal = Double.compare(this.col2Value, o.col2Value);
            }
        }
        return retVal;
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + ((this.col1Name == null) ? 0 : this.col1Name.hashCode());
        long temp;
        temp = Double.doubleToLongBits(this.col1Value);
        result = prime * result + (int) (temp ^ (temp >>> 32));
        result = prime * result + ((this.col2Name == null) ? 0 : this.col2Name.hashCode());
        temp = Double.doubleToLongBits(this.col2Value);
        result = prime * result + (int) (temp ^ (temp >>> 32));
        return result;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (!(obj instanceof CrossReference)) {
            return false;
        }
        CrossReference other = (CrossReference) obj;
        if (this.col1Name == null) {
            if (other.col1Name != null) {
                return false;
            }
        } else if (!this.col1Name.equals(other.col1Name)) {
            return false;
        }
        if (Double.doubleToLongBits(this.col1Value) != Double.doubleToLongBits(other.col1Value)) {
            return false;
        }
        if (this.col2Name == null) {
            if (other.col2Name != null) {
                return false;
            }
        } else if (!this.col2Name.equals(other.col2Name)) {
            return false;
        }
        if (Double.doubleToLongBits(this.col2Value) != Double.doubleToLongBits(other.col2Value)) {
            return false;
        }
        return true;
    }

    @Override
    public String toString() {
        return String.format("%s\t%4.2f\t%s\t%4.2f", this.col1Name, this.col1Value, this.col2Name, this.col2Value);
    }

}
