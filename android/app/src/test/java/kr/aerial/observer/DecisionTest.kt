package kr.aerial.observer

import org.junit.Assert.assertEquals
import org.junit.Test

/** High-risk port boundaries: a changed threshold could turn a bird into a drone alert. */
class DecisionTest {
    @Test fun strictMargin() {
        assertEquals(Decision.UNKNOWN, Decision.label(0,.50f,floatArrayOf(.50f,.30f,0f)))
        assertEquals(Decision.UNKNOWN, Decision.label(0,.40f,floatArrayOf(.40f,.20f,0f)))
        assertEquals("드론 의심", Decision.label(0,.401f,floatArrayOf(.401f,.20f,0f)))
    }
    @Test fun confidenceAndRivalVeto() {
        assertEquals("드론", Decision.label(0,.60f,floatArrayOf(.60f,.10f,.10f)))
        assertEquals("새", Decision.label(1,.60f,floatArrayOf(.10f,.60f,.10f)))
        assertEquals(Decision.UNKNOWN, Decision.label(0,.80f,floatArrayOf(.80f,.31f,0f)))
        assertEquals(Decision.UNKNOWN, Decision.label(0,.80f,floatArrayOf(.80f,.10f,.61f)))
        assertEquals(Decision.UNKNOWN, Decision.label(0,.29f,floatArrayOf(.29f,.01f,.01f)))
    }
    @Test fun invalidScoresDoNotConfirm() {
        assertEquals(Decision.UNKNOWN, Decision.label(0,.80f,floatArrayOf(.10f,.80f,0f)))
        assertEquals(Decision.UNKNOWN, Decision.label(0,.80f,floatArrayOf(.80f,Float.NaN,0f)))
        assertEquals(Decision.UNKNOWN, Decision.label(0,.80f,floatArrayOf(.80f)))
    }
    @Test fun alertLevels() {
        assertEquals(1,Decision.level(Decision.UNKNOWN)); assertEquals(2,Decision.level("드론 의심"))
        assertEquals(3,Decision.level("드론")); assertEquals(0,Decision.level("새"))
        assertEquals(0,Decision.level("관측 끊김"))
    }
}
