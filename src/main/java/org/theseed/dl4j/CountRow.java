/**
 *
 */
package org.theseed.dl4j;

import java.util.BitSet;
import java.util.stream.IntStream;

/**
 * A count row contains a floating-point output value and an array of counters.  Each counter indicates the
 * number of times the corresponding input column had a 1 in it for that output value.  Finally, there is
 * a counter for the total number of records with the output value in them.  Count-rows are sorted by
 * output value, from highest to lowest.
 *
 * @author Bruce Parrello
 *
 */
public class CountRow implements Comparable<CountRow> {

    // FIELDS
    /** output value */
    private double outVal;
    /** number of input records with this output value */
    private int recordCount;
    /** array of counters for each input column */
    private int[] inputCounts;

    /**
     * Construct a new count-row object for the specified number of columns.
     *
     * @param outValue	relevant output value
     * @param length	number of input columns required
     */
    public CountRow(Double outValue, int length) {
        this.outVal = outValue;
        this.recordCount = 0;
        this.inputCounts = new int[length];
    }

    @Override
    public int compareTo(CountRow o) {
        return Double.compare(o.outVal, this.outVal);
    }

    /**
     * Increment the counts in this object.
     *
     * @param goodCols		set of good columns to count
     * @param activeCols	set of columns with a 1 in them
     */
    public void count(BitSet goodCols, BitSet activeCols) {
        // This will be the index in which to count the next good column.
        int idx = 0;
        // Loop through the good columns.
        for (int i = goodCols.nextSetBit(0); i >= 0; i = goodCols.nextSetBit(i+1)) {
            if (activeCols.get(i))
                this.inputCounts[idx]++;
            idx++;
        }
        // Count this record.
        this.recordCount++;
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        long temp;
        temp = Double.doubleToLongBits(this.outVal);
        result = prime * result + (int) (temp ^ (temp >>> 32));
        return result;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (!(obj instanceof CountRow)) {
            return false;
        }
        CountRow other = (CountRow) obj;
        if (Double.doubleToLongBits(this.outVal) != Double.doubleToLongBits(other.outVal)) {
            return false;
        }
        return true;
    }

    /**
     * @return the output value for these counts
     */
    public double getOutVal() {
        return this.outVal;
    }

    /**
     * @return the number of records with this output value
     */
    public int getRecordCount() {
        return this.recordCount;
    }

    /**
     * Accumulate the record counts for this value in the specified count array.
     *
     * @param colCounts		array to be updated
     */
    public void accumulate(int[] colCounts) {
        IntStream.range(0, this.inputCounts.length).forEach(i -> colCounts[i] += this.inputCounts[i]);
    }

}
