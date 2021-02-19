/**
 *
 */
package org.theseed.dl4j;

/**
 * This object represents a parameter impact.  It contains the parameter name and its impact value, and sorts
 * from highest to lowest impact.
 *
 * @author Bruce Parrello
 *
 */
public class Impact implements Comparable<Impact> {

    // FIELDS
    /** name of column */
    private String name;
    /** impact of column */
    private double impact;

    /**
     * Construct an impact object.
     *
     * @param col		name of column
     * @param impact0	impact of column
     */
    public Impact(String col, double impact0) {
        this.name = col;
        this.impact = impact0;
    }

    @Override
    public int compareTo(Impact o) {
        int retVal = Double.compare(o.impact, this.impact);
        if (retVal == 0)
            retVal = this.name.compareTo(o.name);
        return retVal;
    }

    /**
     * @return the name
     */
    public String getName() {
        return this.name;
    }

    /**
     * @return the impact
     */
    public double getImpact() {
        return this.impact;
    }

}
