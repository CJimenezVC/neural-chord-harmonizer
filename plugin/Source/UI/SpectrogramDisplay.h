#pragma once

#include <cmath>
#include <cstring>

#include <juce_gui_basics/juce_gui_basics.h>

/**
    Scrolling, colour spectrogram of the detector's log-frequency feature
    (one row per semitone). New columns scroll in from the right. When a chord
    is detected, the rows belonging to the active pitch classes are highlighted
    and the panel border glows.
*/
class SpectrogramDisplay : public juce::Component
{
public:
    SpectrogramDisplay() { setOpaque (true); }

    /** @p bins = semitone rows, @p cols = history depth; @p midiLo = MIDI of row 0. */
    void configure (int bins, int cols, int midiLo)
    {
        numBins = juce::jmax (1, bins);
        numCols = juce::jmax (1, cols);
        lowMidi = midiLo;
        image = juce::Image (juce::Image::RGB, numCols, numBins, true);
        image.clear (image.getBounds(), colourFor (0.0f));
    }

    /** Push one feature column (log-magnitude per semitone) and scroll left. */
    void pushColumn (const float* vals, int n)
    {
        if (! image.isValid()) return;
        const int rows = juce::jmin (n, numBins);

        juce::Image::BitmapData bmp (image, juce::Image::BitmapData::readWrite);
        const int stride = bmp.pixelStride;
        for (int y = 0; y < numBins; ++y)
        {
            auto* line = bmp.getLinePointer (y);
            std::memmove (line, line + stride, (size_t) (numCols - 1) * (size_t) stride);   // scroll left
            const float v = (y < rows) ? vals[y] : kFloor;
            bmp.setPixelColour (numCols - 1, y, colourFor (norm (v)));
        }
        repaint();
    }

    void setChordMask (int m) noexcept { chordMask = m; }

    void paint (juce::Graphics& g) override
    {
        auto bounds = getLocalBounds().toFloat();
        g.fillAll (juce::Colour (0xff0a0a12));

        if (image.isValid())
        {
            auto inner = bounds.reduced (3.0f);
            g.setImageResamplingQuality (juce::Graphics::mediumResamplingQuality);
            // Flip vertically so low pitch sits at the bottom.
            g.drawImageTransformed (image,
                juce::AffineTransform::verticalFlip ((float) image.getHeight())
                    .scaled (inner.getWidth()  / (float) image.getWidth(),
                             inner.getHeight() / (float) image.getHeight())
                    .translated (inner.getX(), inner.getY()));

            drawChordOverlay (g, inner);
        }

        // Pulsing border: dim when idle, bright cyan/green when a chord is held.
        pulse += 0.08f;
        const bool active = chordMask != 0;
        const float a = active ? (0.55f + 0.45f * std::sin (pulse)) : 0.25f;
        auto edge = active ? juce::Colour (0xff2dffb0) : juce::Colour (0xff39406a);
        g.setColour (edge.withAlpha (a));
        g.drawRoundedRectangle (bounds.reduced (1.5f), 6.0f, active ? 2.5f : 1.5f);
    }

private:
    static constexpr float kFloor = -16.0f;

    float norm (float v) const noexcept
    {
        return juce::jlimit (0.0f, 1.0f, (v - vMin) / (vMax - vMin));
    }

    // Inferno-like ramp: black -> indigo -> magenta -> orange -> pale yellow.
    static juce::Colour colourFor (float t)
    {
        static const float stops[][4] = {
            { 0.00f,   4,   2,  18 }, { 0.25f,  58,  15,  98 },
            { 0.50f, 140,  30, 110 }, { 0.72f, 214,  68,  78 },
            { 0.88f, 246, 132,  32 }, { 1.00f, 252, 255, 168 } };
        t = juce::jlimit (0.0f, 1.0f, t);
        for (int i = 1; i < (int) (sizeof stops / sizeof stops[0]); ++i)
            if (t <= stops[i][0])
            {
                const float* a = stops[i - 1];
                const float* b = stops[i];
                const float f = (t - a[0]) / juce::jmax (1.0e-6f, b[0] - a[0]);
                return juce::Colour::fromRGB (
                    (juce::uint8) (a[1] + f * (b[1] - a[1])),
                    (juce::uint8) (a[2] + f * (b[2] - a[2])),
                    (juce::uint8) (a[3] + f * (b[3] - a[3])));
            }
        return juce::Colour::fromRGB (252, 255, 168);
    }

    void drawChordOverlay (juce::Graphics& g, juce::Rectangle<float> r)
    {
        static const char* names[12] =
            { "C","C#","D","D#","E","F","F#","G","G#","A","A#","B" };
        const float rowH = r.getHeight() / (float) numBins;

        for (int bin = 0; bin < numBins; ++bin)
        {
            const int pc = (lowMidi + bin) % 12;
            if ((chordMask & (1 << pc)) == 0) continue;

            // Flipped: row 0 (low pitch) at the bottom.
            const float y = r.getBottom() - (float) (bin + 1) * rowH;
            g.setColour (juce::Colour (0xff2dffb0).withAlpha (0.18f));
            g.fillRect (r.getX(), y, r.getWidth(), rowH);
        }

        // Note tags for active pitch classes, down the right edge (one per pc).
        g.setFont (juce::Font (12.0f, juce::Font::bold));
        for (int pc = 0; pc < 12; ++pc)
        {
            if ((chordMask & (1 << pc)) == 0) continue;
            // highest occurrence of this pc within range -> a stable label slot
            int topBin = -1;
            for (int bin = 0; bin < numBins; ++bin)
                if ((lowMidi + bin) % 12 == pc) topBin = bin;
            if (topBin < 0) continue;
            const float y = r.getBottom() - (float) (topBin + 1) * rowH;
            juce::Rectangle<float> tag (r.getRight() - 26.0f, y - 7.0f, 24.0f, 16.0f);
            g.setColour (juce::Colour (0xcc101820));
            g.fillRoundedRectangle (tag, 3.0f);
            g.setColour (juce::Colour (0xff2dffb0));
            g.drawText (names[pc], tag, juce::Justification::centred);
        }
    }

    juce::Image image;
    int numBins = 61, numCols = 512, lowMidi = 36;
    int chordMask = 0;
    float vMin = -12.0f, vMax = 2.0f;
    float pulse = 0.0f;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (SpectrogramDisplay)
};
