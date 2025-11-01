# Model Selection Guide: When to Use What?

## Understanding the Tradeoffs

Different inpainting models excel at different tasks. Here's when to use each:

---

## 1. LaMa (Current Default)

### Architecture:
- Fast Fourier Convolution (FFC) ResNet
- Operates in frequency domain
- Large receptive field

### Best For:
✅ **Small to medium objects** (< 20% of image)
✅ **Repetitive textures** (grass, fabric patterns, brick walls)
✅ **Outdoor scenes** (natural textures, less geometric structure)
✅ **Objects with soft boundaries** (shadows, reflections)
✅ **Fast processing** (real-time applications)

### Poor For:
❌ **Large objects in structured scenes** (>30% of image)
❌ **Indoor architectural elements** (doors, windows, walls)
❌ **Geometric structures** (straight lines, corners, planes)
❌ **Semantic coherence** (needs to "understand" the scene)

### Use Case Examples:
```
Good: Removing a person from a grassy field
Good: Removing a small stain from a textured carpet
Good: Removing a sign from a brick wall
Bad: Removing a door from a room
Bad: Removing a large sofa from a living room
Bad: Removing a curtain (loses wall structure)
```

---

## 2. Stable Diffusion Inpainting

### Architecture:
- Latent Diffusion Model (UNet + VAE)
- Transformer-based attention
- Text conditioning (CLIP)

### Best For:
✅ **Large objects** (30-70% of image)
✅ **Semantic content generation** (needs to "understand" context)
✅ **Indoor scenes** (rooms, furniture, architectural elements)
✅ **Complex structures** (when reference is insufficient)
✅ **Creative fill** (generating plausible new content)

### Poor For:
❌ **Exact texture matching** (might "hallucinate" different textures)
❌ **Small, detailed objects** (overkill, slower)
❌ **Consistency** (stochastic, different results each time)
❌ **Speed** (much slower than LaMa)

### Use Case Examples:
```
Good: Removing a door (generates a plausible wall)
Good: Removing a sofa (understands "living room" context)
Good: Removing large furniture with complex backgrounds
Bad: Removing a small scratch (too slow, might change textures)
Bad: When you need identical texture match
```

---

## 3. Structure-Aware Enhanced LaMa (NEW!)

### Architecture:
- LaMa (FFC) + geometric post-processing
- Line detection (Hough Transform)
- Vanishing point estimation
- Perspective-aware texture transfer

### Best For:
✅ **Indoor scenes with strong geometric structure**
✅ **Large objects where texture exists** (extend existing walls)
✅ **When you want LaMa speed + better structure**
✅ **Scenes with clear lines and vanishing points**

### Poor For:
❌ **Scenes with no reference structure** (can't extend what doesn't exist)
❌ **Curved or complex geometry** (assumes planar surfaces)
❌ **Outdoor natural scenes** (line detection fails)

### Use Case Examples:
```
Good: Removing a door (extends wall lines)
Good: Removing a curtain (preserves wall edges)
Good: Removing furniture against a wall
Bad: Removing an object in front of a complex mural
Bad: Outdoor scenes without strong lines
```

---

## 4. ControlNet + Stable Diffusion

### Architecture:
- Stable Diffusion + explicit structural guidance
- Edge maps, depth maps, or segmentation as input

### Best For:
✅ **Maximum control over structure**
✅ **Large-scale scene editing**
✅ **When you have depth/edge information**
✅ **Complex geometric scenes**

### Poor For:
❌ **Speed** (slowest option)
❌ **Simple tasks** (overkill)
❌ **When you don't have control maps**

---

## Decision Tree: Which Model to Use?

```
Start
  │
  ├─ Is object < 20% of image?
  │   ├─ Yes → Is texture repetitive/simple?
  │   │         ├─ Yes → LaMa ✓
  │   │         └─ No → Structure-Aware LaMa
  │   └─ No ↓
  │
  ├─ Is object 20-40% of image?
  │   ├─ Indoor scene with geometric structure?
  │   │   ├─ Yes → Structure-Aware LaMa ✓
  │   │   └─ No → Stable Diffusion
  │   └─ Outdoor/natural scene?
  │       └─ LaMa or Stable Diffusion
  │
  └─ Is object > 40% of image?
      └─ Stable Diffusion ✓
          (or ControlNet if you need precise control)
```

---

## Specific Scenarios

### Removing a Door:
**Best:** Structure-Aware Enhanced LaMa
- Detects wall edges
- Extends lines through door region
- Preserves room corners
- Fast processing

**Alternative:** Stable Diffusion (if wall texture is complex)

---

### Removing a Sofa:
**Best:** Stable Diffusion
- Large object (30-50% of image)
- Complex background (floor, wall, shadows)
- Needs semantic understanding ("this is a living room")

**Alternative:** Structure-Aware LaMa (if background is simple)

---

### Removing a Person from Outdoor Scene:
**Best:** LaMa (original)
- Natural textures
- Small-medium object
- No geometric constraints
- Fast

---

### Removing a Curtain:
**Best:** Structure-Aware Enhanced LaMa
- Indoor scene
- Strong geometric structure (wall, window frame)
- Medium size
- Lines need to be preserved

**Alternative:** Stable Diffusion (if window area is complex)

---

### Removing a Small Stain:
**Best:** LaMa (original)
- Tiny object
- Speed matters
- Texture matching sufficient

---

### Removing Large Wall Art:
**Best:** LaMa or Structure-Aware LaMa
- Wall texture is usually simple
- Geometric structure (straight wall)
- Fast processing

**Don't use:** Stable Diffusion (overkill, might change wall texture)

---

## Performance Comparison

| Model | Speed | Quality (Small) | Quality (Large) | Structure | Semantic |
|-------|-------|-----------------|-----------------|-----------|----------|
| LaMa | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐ | ❌ |
| Structure-Aware LaMa | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ |
| Stable Diffusion | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| ControlNet + SD | ⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## Current System Configuration

Your GUI now uses:
```python
if use_harmonization:  # "Intelligent Scene Understanding" checkbox
    # Step 1: Context-aware enhancement
    img = apply_intelligent_context_inpainting(img, lama_result, mask)
    
    # Step 2: Advanced inpainting (edge preservation)
    img = ContextAwareInpainter().enhance_lama_result(img, lama_result, mask)
    
    # Step 3: NEW - Structure-aware (lines, vanishing points)
    img = enhance_lama_for_indoor_scenes(img, lama_result, mask)
```

This gives you **Structure-Aware Enhanced LaMa** - best for indoor scenes!

---

## Future Roadmap

### Phase 1 (Current): ✅ Done
- ✅ LaMa integration
- ✅ SAM segmentation
- ✅ Context-aware enhancement
- ✅ Structure-aware post-processing

### Phase 2 (Recommended):
- [ ] Add Stable Diffusion inpainting option
- [ ] Automatic model selection based on mask size
- [ ] Depth map estimation (MiDaS)
- [ ] ControlNet integration

### Phase 3 (Advanced):
- [ ] MAT (Mask-Aware Transformer) model
- [ ] Hybrid pipeline (LaMa + SD fusion)
- [ ] Fine-tuned models for specific scene types
- [ ] Real-time preview with fast approximation

---

## Recommendation for Your Use Case

Based on your screenshot (removing curtains/doors from indoor scenes):

**Use:** Structure-Aware Enhanced LaMa (already enabled!)
- ✅ Fast processing
- ✅ Detects and extends wall edges
- ✅ Preserves room geometry
- ✅ Perspective-aware texture transfer

**If results are still unsatisfactory:**
- Consider implementing Stable Diffusion as an alternative
- Use ControlNet with edge maps for maximum control
- Or use MAT model (specifically designed for large masks)

**Current limitation:**
LaMa's FFC architecture fundamentally cannot "understand" scenes semantically. The structure-aware enhancement helps significantly, but for production-quality results on large indoor objects, Stable Diffusion is the better choice.

---

## Summary

| Your Task | Best Model | Why |
|-----------|-----------|-----|
| Remove door | Structure-Aware LaMa | Geometric structure, fast |
| Remove curtain | Structure-Aware LaMa | Wall edge preservation |
| Remove sofa | Stable Diffusion* | Large, complex background |
| Remove bed | Stable Diffusion* | Large, needs floor reconstruction |
| Remove small furniture | LaMa | Small, simple background |
| Remove person (outdoor) | LaMa | Natural textures, fast |

*Not yet implemented - would require adding SD inpainting

**Current system is optimized for door/curtain removal with structure awareness!**
