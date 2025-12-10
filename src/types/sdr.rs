//! Sparse Distributed Representation (SDR) implementation.
//!
//! An SDR is a data structure representing a group of boolean values (bits).
//! It can be represented in three formats:
//! - **Dense**: A contiguous array of all bits
//! - **Sparse**: A sorted list of indices of active (true) bits
//! - **Coordinate**: A list of coordinates for each dimension
//!
//! The SDR automatically converts between formats and caches results for efficiency.

use crate::error::{MokoshError, Result};
use crate::types::{ElemDense, ElemSparse, Real, UInt};
use crate::utils::Random;

use std::cell::RefCell;
use std::fmt;

/// Type alias for dense SDR data (array of bytes, 0 or 1).
pub type SdrDense = Vec<ElemDense>;

/// Type alias for sparse SDR data (sorted indices of active bits).
pub type SdrSparse = Vec<ElemSparse>;

/// Type alias for coordinate SDR data (coordinates per dimension).
pub type SdrCoordinate = Vec<Vec<UInt>>;

/// Callback function type for SDR value changes.
pub type SdrCallback = Box<dyn Fn() + Send + Sync>;

/// Internal cache state for lazy evaluation.
#[derive(Default)]
struct SdrCache {
    dense: Option<SdrDense>,
    sparse: Option<SdrSparse>,
    coordinates: Option<SdrCoordinate>,
}

/// Sparse Distributed Representation.
///
/// This is the fundamental data structure in HTM. It represents a binary vector
/// where typically only a small percentage of bits are active (true).
///
/// # Example
///
/// ```rust
/// use mokosh::types::Sdr;
///
/// // Create a 10x10 SDR
/// let mut sdr = Sdr::new(&[10, 10]);
///
/// // Set active bits using sparse indices
/// sdr.set_sparse(&[1, 4, 8, 15, 42]).unwrap();
///
/// // Get the number of active bits
/// assert_eq!(sdr.get_sum(), 5);
///
/// // Access in different formats
/// let dense = sdr.get_dense();
/// let sparse = sdr.get_sparse();
/// let coords = sdr.get_coordinates();
/// ```
pub struct Sdr {
    /// Dimensions of the SDR.
    dimensions: Vec<UInt>,

    /// Total size (product of dimensions).
    size: usize,

    /// Cached representations (interior mutability for lazy evaluation).
    cache: RefCell<SdrCache>,

    /// Callbacks to notify on value changes.
    callbacks: RefCell<Vec<Option<SdrCallback>>>,

    /// Callbacks to notify on destruction.
    destroy_callbacks: RefCell<Vec<Option<SdrCallback>>>,
}

// Custom serialization for Sdr - we serialize dimensions and sparse indices.
#[cfg(feature = "serde")]
mod serde_impl {
    use super::*;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    #[derive(Serialize, Deserialize)]
    struct SdrState {
        dimensions: Vec<UInt>,
        sparse: Vec<ElemSparse>,
    }

    impl Serialize for Sdr {
        fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            let state = SdrState {
                dimensions: self.dimensions.clone(),
                sparse: self.get_sparse().to_vec(),
            };
            state.serialize(serializer)
        }
    }

    impl<'de> Deserialize<'de> for Sdr {
        fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
        where
            D: Deserializer<'de>,
        {
            let state = SdrState::deserialize(deserializer)?;
            let mut sdr = Sdr::new(&state.dimensions);
            sdr.set_sparse_unchecked(state.sparse);
            Ok(sdr)
        }
    }
}

impl Sdr {
    /// Creates a new SDR with the given dimensions, initialized to all zeros.
    ///
    /// # Arguments
    ///
    /// * `dimensions` - The shape of the SDR (e.g., `&[10, 10]` for 10x10)
    ///
    /// # Panics
    ///
    /// Panics if dimensions is empty or contains zeros.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mokosh::types::Sdr;
    ///
    /// let sdr = Sdr::new(&[100]);        // 1D SDR with 100 bits
    /// let sdr2 = Sdr::new(&[10, 10]);    // 2D SDR with 100 bits
    /// let sdr3 = Sdr::new(&[5, 4, 5]);   // 3D SDR with 100 bits
    /// ```
    #[must_use]
    pub fn new(dimensions: &[UInt]) -> Self {
        assert!(!dimensions.is_empty(), "Dimensions cannot be empty");

        let size: usize = dimensions.iter().map(|&d| d as usize).product();

        // Allow size 0 for placeholder SDRs
        for (i, &dim) in dimensions.iter().enumerate() {
            if dim == 0 && dimensions.len() > 1 {
                panic!("Dimension {} cannot be zero in multi-dimensional SDR", i);
            }
        }

        Self {
            dimensions: dimensions.to_vec(),
            size,
            cache: RefCell::new(SdrCache::default()),
            callbacks: RefCell::new(Vec::new()),
            destroy_callbacks: RefCell::new(Vec::new()),
        }
    }

    /// Creates a new SDR with dimensions initialized from an iterator.
    pub fn with_dimensions<I>(dimensions: I) -> Self
    where
        I: IntoIterator<Item = UInt>,
    {
        let dims: Vec<UInt> = dimensions.into_iter().collect();
        Self::new(&dims)
    }

    /// Returns the dimensions of this SDR.
    #[inline]
    #[must_use]
    pub fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    /// Returns the total number of bits in the SDR.
    #[inline]
    #[must_use]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Returns the number of dimensions.
    #[inline]
    #[must_use]
    pub fn num_dimensions(&self) -> usize {
        self.dimensions.len()
    }

    /// Reshapes the SDR to new dimensions. The total size must remain the same.
    ///
    /// # Errors
    ///
    /// Returns an error if the new dimensions have a different total size.
    pub fn reshape(&mut self, new_dimensions: &[UInt]) -> Result<()> {
        let new_size: usize = new_dimensions.iter().map(|&d| d as usize).product();

        if new_size != self.size {
            return Err(MokoshError::InvalidDimensions(format!(
                "Cannot reshape from size {} to size {}",
                self.size, new_size
            )));
        }

        self.dimensions = new_dimensions.to_vec();

        // Invalidate coordinate cache as it depends on dimensions
        self.cache.borrow_mut().coordinates = None;

        Ok(())
    }

    /// Sets all bits to zero.
    pub fn zero(&mut self) {
        let mut cache = self.cache.borrow_mut();
        cache.dense = Some(vec![0; self.size]);
        cache.sparse = Some(Vec::new());
        cache.coordinates = Some(vec![Vec::new(); self.dimensions.len()]);
        drop(cache);

        self.do_callbacks();
    }

    /// Clears all cached representations.
    fn clear_cache(&self) {
        let mut cache = self.cache.borrow_mut();
        cache.dense = None;
        cache.sparse = None;
        cache.coordinates = None;
    }

    /// Invokes all registered callbacks.
    fn do_callbacks(&self) {
        let callbacks = self.callbacks.borrow();
        for callback in callbacks.iter().flatten() {
            callback();
        }
    }

    // ========================================================================
    // Dense format operations
    // ========================================================================

    /// Sets the SDR value from a dense array.
    ///
    /// # Arguments
    ///
    /// * `data` - A slice of values where non-zero means active
    ///
    /// # Errors
    ///
    /// Returns an error if the data length doesn't match the SDR size.
    pub fn set_dense(&mut self, data: &[ElemDense]) -> Result<()> {
        if data.len() != self.size {
            return Err(MokoshError::DimensionMismatch {
                expected: vec![self.size as u32],
                actual: vec![data.len() as u32],
            });
        }

        let mut cache = self.cache.borrow_mut();
        cache.dense = Some(data.to_vec());
        cache.sparse = None;
        cache.coordinates = None;
        drop(cache);

        self.do_callbacks();
        Ok(())
    }

    /// Sets the SDR value from a dense array, consuming it to avoid copying.
    pub fn set_dense_owned(&mut self, data: SdrDense) -> Result<()> {
        if data.len() != self.size {
            return Err(MokoshError::DimensionMismatch {
                expected: vec![self.size as u32],
                actual: vec![data.len() as u32],
            });
        }

        let mut cache = self.cache.borrow_mut();
        cache.dense = Some(data);
        cache.sparse = None;
        cache.coordinates = None;
        drop(cache);

        self.do_callbacks();
        Ok(())
    }

    /// Gets the dense representation of the SDR.
    ///
    /// This method lazily computes the dense array from sparse or coordinate
    /// representations if needed.
    #[must_use]
    pub fn get_dense(&self) -> SdrDense {
        {
            let cache = self.cache.borrow();
            if let Some(ref dense) = cache.dense {
                return dense.clone();
            }
        }

        // Need to compute from sparse
        let sparse = self.get_sparse();
        let mut dense = vec![0u8; self.size];
        for &idx in &sparse {
            dense[idx as usize] = 1;
        }

        let mut cache = self.cache.borrow_mut();
        cache.dense = Some(dense.clone());
        dense
    }

    /// Gets a reference to the dense representation, computing if necessary.
    pub fn with_dense<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&SdrDense) -> R,
    {
        // Ensure dense is computed
        {
            let cache = self.cache.borrow();
            if cache.dense.is_some() {
                return f(cache.dense.as_ref().unwrap());
            }
        }

        let _ = self.get_dense();
        let cache = self.cache.borrow();
        f(cache.dense.as_ref().unwrap())
    }

    // ========================================================================
    // Sparse format operations
    // ========================================================================

    /// Sets the SDR value from sparse indices.
    ///
    /// # Arguments
    ///
    /// * `indices` - Sorted slice of indices of active bits
    ///
    /// # Errors
    ///
    /// Returns an error if indices are not sorted, contain duplicates, or are out of bounds.
    pub fn set_sparse(&mut self, indices: &[ElemSparse]) -> Result<()> {
        // Validate indices
        self.validate_sparse(indices)?;

        let mut cache = self.cache.borrow_mut();
        cache.sparse = Some(indices.to_vec());
        cache.dense = None;
        cache.coordinates = None;
        drop(cache);

        self.do_callbacks();
        Ok(())
    }

    /// Sets the SDR value from sparse indices, consuming to avoid copying.
    pub fn set_sparse_owned(&mut self, indices: SdrSparse) -> Result<()> {
        self.validate_sparse(&indices)?;

        let mut cache = self.cache.borrow_mut();
        cache.sparse = Some(indices);
        cache.dense = None;
        cache.coordinates = None;
        drop(cache);

        self.do_callbacks();
        Ok(())
    }

    /// Sets sparse indices without validation (for internal use).
    pub(crate) fn set_sparse_unchecked(&mut self, indices: SdrSparse) {
        let mut cache = self.cache.borrow_mut();
        cache.sparse = Some(indices);
        cache.dense = None;
        cache.coordinates = None;
        drop(cache);

        self.do_callbacks();
    }

    /// Validates sparse indices.
    fn validate_sparse(&self, indices: &[ElemSparse]) -> Result<()> {
        if indices.is_empty() {
            return Ok(());
        }

        // Check bounds and ordering
        let mut prev = indices[0];
        if prev as usize >= self.size {
            return Err(MokoshError::IndexOutOfBounds {
                index: prev as usize,
                size: self.size,
            });
        }

        for &idx in &indices[1..] {
            if idx <= prev {
                return Err(MokoshError::InvalidSdrData(
                    "Sparse indices must be sorted and unique".to_string(),
                ));
            }
            if idx as usize >= self.size {
                return Err(MokoshError::IndexOutOfBounds {
                    index: idx as usize,
                    size: self.size,
                });
            }
            prev = idx;
        }

        Ok(())
    }

    /// Gets the sparse representation of the SDR.
    #[must_use]
    pub fn get_sparse(&self) -> SdrSparse {
        {
            let cache = self.cache.borrow();
            if let Some(ref sparse) = cache.sparse {
                return sparse.clone();
            }
        }

        // Compute from dense or coordinates
        let sparse = {
            let cache = self.cache.borrow();
            if let Some(ref dense) = cache.dense {
                dense
                    .iter()
                    .enumerate()
                    .filter(|(_, &v)| v != 0)
                    .map(|(i, _)| i as ElemSparse)
                    .collect()
            } else if let Some(ref coords) = cache.coordinates {
                self.coordinates_to_sparse(coords)
            } else {
                // No data set, return empty
                Vec::new()
            }
        };

        let mut cache = self.cache.borrow_mut();
        cache.sparse = Some(sparse.clone());
        sparse
    }

    /// Gets a reference to the sparse representation.
    pub fn with_sparse<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&SdrSparse) -> R,
    {
        {
            let cache = self.cache.borrow();
            if cache.sparse.is_some() {
                return f(cache.sparse.as_ref().unwrap());
            }
        }

        let _ = self.get_sparse();
        let cache = self.cache.borrow();
        f(cache.sparse.as_ref().unwrap())
    }

    // ========================================================================
    // Coordinate format operations
    // ========================================================================

    /// Sets the SDR value from coordinates.
    ///
    /// # Arguments
    ///
    /// * `coordinates` - A vector of coordinate vectors, one per dimension
    ///
    /// # Errors
    ///
    /// Returns an error if coordinates are invalid.
    pub fn set_coordinates(&mut self, coordinates: &SdrCoordinate) -> Result<()> {
        if coordinates.len() != self.dimensions.len() {
            return Err(MokoshError::InvalidDimensions(format!(
                "Expected {} dimensions, got {}",
                self.dimensions.len(),
                coordinates.len()
            )));
        }

        // Validate that all inner vectors have the same length
        if !coordinates.is_empty() {
            let len = coordinates[0].len();
            for (i, coord) in coordinates.iter().enumerate() {
                if coord.len() != len {
                    return Err(MokoshError::InvalidSdrData(format!(
                        "Coordinate dimension {} has length {}, expected {}",
                        i,
                        coord.len(),
                        len
                    )));
                }
            }
        }

        // Validate bounds
        for (dim_idx, (coords, &dim_size)) in coordinates.iter().zip(&self.dimensions).enumerate() {
            for &c in coords {
                if c >= dim_size {
                    return Err(MokoshError::IndexOutOfBounds {
                        index: c as usize,
                        size: dim_size as usize,
                    });
                }
            }
        }

        let mut cache = self.cache.borrow_mut();
        cache.coordinates = Some(coordinates.clone());
        cache.dense = None;
        cache.sparse = None;
        drop(cache);

        self.do_callbacks();
        Ok(())
    }

    /// Gets the coordinate representation of the SDR.
    #[must_use]
    pub fn get_coordinates(&self) -> SdrCoordinate {
        {
            let cache = self.cache.borrow();
            if let Some(ref coords) = cache.coordinates {
                return coords.clone();
            }
        }

        // Compute from sparse
        let sparse = self.get_sparse();
        let coords = self.sparse_to_coordinates(&sparse);

        let mut cache = self.cache.borrow_mut();
        cache.coordinates = Some(coords.clone());
        coords
    }

    /// Converts flat indices to coordinates.
    fn sparse_to_coordinates(&self, sparse: &[ElemSparse]) -> SdrCoordinate {
        let num_dims = self.dimensions.len();
        let mut coordinates: SdrCoordinate = vec![Vec::with_capacity(sparse.len()); num_dims];

        for &flat_idx in sparse {
            let mut idx = flat_idx as usize;
            for dim in (0..num_dims).rev() {
                let dim_size = self.dimensions[dim] as usize;
                coordinates[dim].push((idx % dim_size) as UInt);
                idx /= dim_size;
            }
        }

        // Reverse each dimension's coordinates since we computed them backwards
        for coords in &mut coordinates {
            coords.reverse();
            // Re-reverse to maintain original order
        }

        // Actually, let me reconsider - the coordinates should be in the same
        // order as the sparse indices. Let me fix this.
        let mut coordinates: SdrCoordinate = vec![Vec::with_capacity(sparse.len()); num_dims];

        for &flat_idx in sparse {
            let mut idx = flat_idx as usize;
            let mut temp_coords = vec![0u32; num_dims];

            for dim in (0..num_dims).rev() {
                let dim_size = self.dimensions[dim] as usize;
                temp_coords[dim] = (idx % dim_size) as UInt;
                idx /= dim_size;
            }

            for (dim, &coord) in temp_coords.iter().enumerate() {
                coordinates[dim].push(coord);
            }
        }

        coordinates
    }

    /// Converts coordinates to flat indices.
    fn coordinates_to_sparse(&self, coordinates: &SdrCoordinate) -> SdrSparse {
        if coordinates.is_empty() || coordinates[0].is_empty() {
            return Vec::new();
        }

        let num_points = coordinates[0].len();
        let mut sparse = Vec::with_capacity(num_points);

        for i in 0..num_points {
            let mut flat_idx: usize = 0;
            let mut multiplier: usize = 1;

            for dim in (0..self.dimensions.len()).rev() {
                flat_idx += coordinates[dim][i] as usize * multiplier;
                multiplier *= self.dimensions[dim] as usize;
            }

            sparse.push(flat_idx as ElemSparse);
        }

        // Sort and deduplicate
        sparse.sort_unstable();
        sparse.dedup();
        sparse
    }

    // ========================================================================
    // Value queries
    // ========================================================================

    /// Returns the value at the given coordinates.
    #[must_use]
    pub fn at(&self, coordinates: &[UInt]) -> bool {
        assert_eq!(
            coordinates.len(),
            self.dimensions.len(),
            "Coordinate dimensions mismatch"
        );

        let flat_idx = self.coordinates_to_flat(coordinates);
        self.with_dense(|dense| dense[flat_idx] != 0)
    }

    /// Converts coordinates to a flat index.
    fn coordinates_to_flat(&self, coordinates: &[UInt]) -> usize {
        let mut flat_idx: usize = 0;
        let mut multiplier: usize = 1;

        for dim in (0..self.dimensions.len()).rev() {
            flat_idx += coordinates[dim] as usize * multiplier;
            multiplier *= self.dimensions[dim] as usize;
        }

        flat_idx
    }

    /// Returns the number of active (true) bits.
    #[must_use]
    pub fn get_sum(&self) -> usize {
        self.with_sparse(Vec::len)
    }

    /// Returns the sparsity (fraction of active bits).
    #[must_use]
    pub fn get_sparsity(&self) -> Real {
        if self.size == 0 {
            return 0.0;
        }
        self.get_sum() as Real / self.size as Real
    }

    /// Returns the number of bits that are active in both SDRs.
    #[must_use]
    pub fn get_overlap(&self, other: &Sdr) -> usize {
        let a = self.get_sparse();
        let b = other.get_sparse();

        // Use set intersection for sorted vectors
        let mut count = 0;
        let mut i = 0;
        let mut j = 0;

        while i < a.len() && j < b.len() {
            match a[i].cmp(&b[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    count += 1;
                    i += 1;
                    j += 1;
                }
            }
        }

        count
    }

    // ========================================================================
    // SDR operations
    // ========================================================================

    /// Copies the value from another SDR.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions don't match.
    pub fn set_sdr(&mut self, other: &Sdr) -> Result<()> {
        if self.dimensions != other.dimensions {
            return Err(MokoshError::DimensionMismatch {
                expected: self.dimensions.clone(),
                actual: other.dimensions.clone(),
            });
        }

        let sparse = other.get_sparse();
        self.set_sparse_owned(sparse)
    }

    /// Randomizes the SDR with the given sparsity.
    ///
    /// # Arguments
    ///
    /// * `sparsity` - Fraction of bits to set active (0.0 to 1.0)
    /// * `rng` - Random number generator
    pub fn randomize(&mut self, sparsity: Real, rng: &mut Random) {
        let num_active = ((self.size as Real) * sparsity).round() as usize;

        if num_active == 0 {
            self.zero();
            return;
        }

        if num_active >= self.size {
            let mut cache = self.cache.borrow_mut();
            cache.dense = Some(vec![1; self.size]);
            cache.sparse = Some((0..self.size as ElemSparse).collect());
            cache.coordinates = None;
            drop(cache);
            self.do_callbacks();
            return;
        }

        // Generate random indices
        let indices = rng.sample((0..self.size as ElemSparse).collect(), num_active);
        let mut sparse: SdrSparse = indices;
        sparse.sort_unstable();

        let mut cache = self.cache.borrow_mut();
        cache.sparse = Some(sparse);
        cache.dense = None;
        cache.coordinates = None;
        drop(cache);

        self.do_callbacks();
    }

    /// Adds noise to the SDR by flipping a fraction of bits.
    ///
    /// # Arguments
    ///
    /// * `fraction_noise` - Fraction of active bits to move (0.0 to 1.0)
    /// * `rng` - Random number generator
    pub fn add_noise(&mut self, fraction_noise: Real, rng: &mut Random) {
        let sparse = self.get_sparse();
        let num_active = sparse.len();

        if num_active == 0 || fraction_noise <= 0.0 {
            return;
        }

        let num_to_flip = ((num_active as Real) * fraction_noise).round() as usize;
        if num_to_flip == 0 {
            return;
        }

        // Select bits to turn off
        let turn_off = rng.sample(sparse.clone(), num_to_flip);

        // Find inactive bits to turn on
        let active_set: std::collections::HashSet<_> = sparse.iter().copied().collect();
        let inactive: Vec<ElemSparse> = (0..self.size as ElemSparse)
            .filter(|&i| !active_set.contains(&i))
            .collect();

        let turn_on = rng.sample(inactive, num_to_flip);

        // Create new sparse representation
        let turn_off_set: std::collections::HashSet<_> = turn_off.iter().copied().collect();
        let mut new_sparse: SdrSparse = sparse
            .into_iter()
            .filter(|&i| !turn_off_set.contains(&i))
            .chain(turn_on)
            .collect();
        new_sparse.sort_unstable();

        let mut cache = self.cache.borrow_mut();
        cache.sparse = Some(new_sparse);
        cache.dense = None;
        cache.coordinates = None;
        drop(cache);

        self.do_callbacks();
    }

    /// Computes the intersection of two SDRs into this SDR.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions don't match.
    pub fn intersection(&mut self, a: &Sdr, b: &Sdr) -> Result<()> {
        if a.dimensions != b.dimensions {
            return Err(MokoshError::DimensionMismatch {
                expected: a.dimensions.clone(),
                actual: b.dimensions.clone(),
            });
        }

        if self.dimensions != a.dimensions {
            return Err(MokoshError::DimensionMismatch {
                expected: self.dimensions.clone(),
                actual: a.dimensions.clone(),
            });
        }

        let sparse_a = a.get_sparse();
        let sparse_b = b.get_sparse();

        // Set intersection of sorted vectors
        let mut result = Vec::new();
        let mut i = 0;
        let mut j = 0;

        while i < sparse_a.len() && j < sparse_b.len() {
            match sparse_a[i].cmp(&sparse_b[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    result.push(sparse_a[i]);
                    i += 1;
                    j += 1;
                }
            }
        }

        self.set_sparse_unchecked(result);
        Ok(())
    }

    /// Computes the union of two SDRs into this SDR.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions don't match.
    pub fn set_union(&mut self, a: &Sdr, b: &Sdr) -> Result<()> {
        if a.dimensions != b.dimensions {
            return Err(MokoshError::DimensionMismatch {
                expected: a.dimensions.clone(),
                actual: b.dimensions.clone(),
            });
        }

        if self.dimensions != a.dimensions {
            return Err(MokoshError::DimensionMismatch {
                expected: self.dimensions.clone(),
                actual: a.dimensions.clone(),
            });
        }

        let sparse_a = a.get_sparse();
        let sparse_b = b.get_sparse();

        // Set union of sorted vectors
        let mut result = Vec::with_capacity(sparse_a.len() + sparse_b.len());
        let mut i = 0;
        let mut j = 0;

        while i < sparse_a.len() && j < sparse_b.len() {
            match sparse_a[i].cmp(&sparse_b[j]) {
                std::cmp::Ordering::Less => {
                    result.push(sparse_a[i]);
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    result.push(sparse_b[j]);
                    j += 1;
                }
                std::cmp::Ordering::Equal => {
                    result.push(sparse_a[i]);
                    i += 1;
                    j += 1;
                }
            }
        }

        result.extend(&sparse_a[i..]);
        result.extend(&sparse_b[j..]);

        self.set_sparse_unchecked(result);
        Ok(())
    }

    /// Concatenates SDRs along an axis into this SDR.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions don't match.
    pub fn concatenate(&mut self, inputs: &[&Sdr], axis: usize) -> Result<()> {
        if inputs.is_empty() {
            return Err(MokoshError::InvalidParameter {
                name: "inputs",
                message: "Cannot concatenate empty list".to_string(),
            });
        }

        // Verify all inputs have compatible dimensions
        let num_dims = inputs[0].num_dimensions();
        for (i, input) in inputs.iter().enumerate() {
            if input.num_dimensions() != num_dims {
                return Err(MokoshError::InvalidDimensions(format!(
                    "Input {} has {} dimensions, expected {}",
                    i,
                    input.num_dimensions(),
                    num_dims
                )));
            }
        }

        // Compute output sparse representation
        let mut result = Vec::new();
        let mut offset: usize = 0;

        for input in inputs {
            let sparse = input.get_sparse();
            for &idx in &sparse {
                result.push((idx as usize + offset) as ElemSparse);
            }
            offset += input.size();
        }

        // Verify result fits in this SDR
        if offset != self.size {
            return Err(MokoshError::DimensionMismatch {
                expected: vec![self.size as u32],
                actual: vec![offset as u32],
            });
        }

        self.set_sparse_unchecked(result);
        Ok(())
    }

    // ========================================================================
    // Callbacks
    // ========================================================================

    /// Adds a callback that is called whenever the SDR value changes.
    ///
    /// Returns a handle that can be used to remove the callback.
    pub fn add_callback(&self, callback: SdrCallback) -> usize {
        let mut callbacks = self.callbacks.borrow_mut();
        let handle = callbacks.len();
        callbacks.push(Some(callback));
        handle
    }

    /// Removes a callback by its handle.
    pub fn remove_callback(&self, handle: usize) -> Result<()> {
        let mut callbacks = self.callbacks.borrow_mut();
        if handle >= callbacks.len() || callbacks[handle].is_none() {
            return Err(MokoshError::InvalidParameter {
                name: "handle",
                message: format!("Invalid callback handle: {}", handle),
            });
        }
        callbacks[handle] = None;
        Ok(())
    }

    /// Adds a callback that is called when the SDR is destroyed.
    pub fn add_destroy_callback(&self, callback: SdrCallback) -> usize {
        let mut callbacks = self.destroy_callbacks.borrow_mut();
        let handle = callbacks.len();
        callbacks.push(Some(callback));
        handle
    }

    /// Removes a destroy callback by its handle.
    pub fn remove_destroy_callback(&self, handle: usize) -> Result<()> {
        let mut callbacks = self.destroy_callbacks.borrow_mut();
        if handle >= callbacks.len() || callbacks[handle].is_none() {
            return Err(MokoshError::InvalidParameter {
                name: "handle",
                message: format!("Invalid destroy callback handle: {}", handle),
            });
        }
        callbacks[handle] = None;
        Ok(())
    }
}

impl Clone for Sdr {
    fn clone(&self) -> Self {
        let mut new_sdr = Self::new(&self.dimensions);

        // Copy the most efficient representation available
        let cache = self.cache.borrow();
        if let Some(ref sparse) = cache.sparse {
            new_sdr.cache.borrow_mut().sparse = Some(sparse.clone());
        } else if let Some(ref dense) = cache.dense {
            new_sdr.cache.borrow_mut().dense = Some(dense.clone());
        } else if let Some(ref coords) = cache.coordinates {
            new_sdr.cache.borrow_mut().coordinates = Some(coords.clone());
        }

        new_sdr
    }
}

impl PartialEq for Sdr {
    fn eq(&self, other: &Self) -> bool {
        if self.dimensions != other.dimensions {
            return false;
        }
        self.get_sparse() == other.get_sparse()
    }
}

impl Eq for Sdr {}

impl fmt::Debug for Sdr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let sparse = self.get_sparse();
        write!(f, "SDR({:?}) {:?}", self.dimensions, sparse)
    }
}

impl fmt::Display for Sdr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SDR( ")?;
        for (i, dim) in self.dimensions.iter().enumerate() {
            write!(f, "{}", dim)?;
            if i + 1 != self.dimensions.len() {
                write!(f, ", ")?;
            }
        }
        write!(f, " ) ")?;

        let sparse = self.get_sparse();
        for (i, &idx) in sparse.iter().enumerate() {
            write!(f, "{}", idx)?;
            if i + 1 != sparse.len() {
                write!(f, ", ")?;
            }
        }
        Ok(())
    }
}

impl Drop for Sdr {
    fn drop(&mut self) {
        let callbacks = self.destroy_callbacks.borrow();
        for callback in callbacks.iter().flatten() {
            callback();
        }
    }
}

impl Default for Sdr {
    fn default() -> Self {
        Self::new(&[0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constructor() {
        let sdr = Sdr::new(&[3]);
        assert_eq!(sdr.size(), 3);
        assert_eq!(sdr.dimensions(), &[3]);

        let sdr2 = Sdr::new(&[3, 4, 5]);
        assert_eq!(sdr2.size(), 60);
        assert_eq!(sdr2.dimensions(), &[3, 4, 5]);
    }

    #[test]
    fn test_empty_placeholder() {
        let sdr = Sdr::new(&[0]);
        assert_eq!(sdr.size(), 0);
    }

    #[test]
    fn test_zero() {
        let mut sdr = Sdr::new(&[4, 4]);
        sdr.set_dense(&vec![1; 16]).unwrap();
        sdr.zero();
        assert_eq!(sdr.get_sum(), 0);
    }

    #[test]
    fn test_dense_sparse_conversion() {
        let mut sdr = Sdr::new(&[9]);
        sdr.set_dense(&[0, 1, 0, 0, 1, 0, 0, 0, 1]).unwrap();
        assert_eq!(sdr.get_sparse(), vec![1, 4, 8]);

        sdr.set_sparse(&[1, 4, 8]).unwrap();
        assert_eq!(sdr.get_dense(), vec![0, 1, 0, 0, 1, 0, 0, 0, 1]);
    }

    #[test]
    fn test_coordinates() {
        let mut sdr = Sdr::new(&[3, 3]);
        sdr.set_coordinates(&vec![vec![0, 1, 2], vec![1, 1, 2]]).unwrap();
        assert_eq!(sdr.get_sparse(), vec![1, 4, 8]);

        sdr.set_sparse(&[1, 4, 8]).unwrap();
        let coords = sdr.get_coordinates();
        assert_eq!(coords, vec![vec![0, 1, 2], vec![1, 1, 2]]);
    }

    #[test]
    fn test_at() {
        let mut sdr = Sdr::new(&[3, 3]);
        sdr.set_sparse(&[4, 5, 8]).unwrap();
        assert!(sdr.at(&[1, 1]));
        assert!(sdr.at(&[1, 2]));
        assert!(sdr.at(&[2, 2]));
        assert!(!sdr.at(&[0, 0]));
    }

    #[test]
    fn test_sum_sparsity() {
        let mut sdr = Sdr::new(&[100]);
        sdr.set_sparse(&[1, 2, 3, 4, 5]).unwrap();
        assert_eq!(sdr.get_sum(), 5);
        assert!((sdr.get_sparsity() - 0.05).abs() < 0.001);
    }

    #[test]
    fn test_overlap() {
        let mut a = Sdr::new(&[9]);
        let mut b = Sdr::new(&[9]);
        a.set_sparse(&[1, 2, 3, 4]).unwrap();
        b.set_sparse(&[2, 3, 4, 5]).unwrap();
        assert_eq!(a.get_overlap(&b), 3);
    }

    #[test]
    fn test_intersection() {
        let mut a = Sdr::new(&[10]);
        let mut b = Sdr::new(&[10]);
        let mut c = Sdr::new(&[10]);

        a.set_sparse(&[0, 1, 2, 3]).unwrap();
        b.set_sparse(&[2, 3, 4, 5]).unwrap();
        c.intersection(&a, &b).unwrap();

        assert_eq!(c.get_sparse(), vec![2, 3]);
    }

    #[test]
    fn test_union() {
        let mut a = Sdr::new(&[10]);
        let mut b = Sdr::new(&[10]);
        let mut c = Sdr::new(&[10]);

        a.set_sparse(&[0, 1, 2, 3]).unwrap();
        b.set_sparse(&[2, 3, 4, 5]).unwrap();
        c.set_union(&a, &b).unwrap();

        assert_eq!(c.get_sparse(), vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_concatenate() {
        let mut a = Sdr::new(&[10]);
        let mut b = Sdr::new(&[10]);
        let mut c = Sdr::new(&[20]);

        a.set_sparse(&[0, 1, 2]).unwrap();
        b.set_sparse(&[0, 1, 2]).unwrap();
        c.concatenate(&[&a, &b], 0).unwrap();

        assert_eq!(c.get_sparse(), vec![0, 1, 2, 10, 11, 12]);
    }

    #[test]
    fn test_equality() {
        let mut a = Sdr::new(&[10]);
        let mut b = Sdr::new(&[10]);

        a.set_sparse(&[1, 2, 3]).unwrap();
        b.set_sparse(&[1, 2, 3]).unwrap();
        assert_eq!(a, b);

        b.set_sparse(&[1, 2, 4]).unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn test_reshape() {
        let mut sdr = Sdr::new(&[3, 4, 5]);
        sdr.set_sparse(&[0, 5, 10]).unwrap();

        sdr.reshape(&[5, 12]).unwrap();
        assert_eq!(sdr.dimensions(), &[5, 12]);
        assert_eq!(sdr.get_sparse(), vec![0, 5, 10]);
    }

    #[test]
    fn test_display() {
        let mut sdr = Sdr::new(&[3, 3]);
        sdr.set_sparse(&[1, 4, 8]).unwrap();
        let s = format!("{}", sdr);
        assert!(s.contains("SDR( 3, 3 )"));
        assert!(s.contains("1, 4, 8"));
    }

    #[test]
    fn test_clone() {
        let mut sdr = Sdr::new(&[10]);
        sdr.set_sparse(&[1, 2, 3]).unwrap();

        let cloned = sdr.clone();
        assert_eq!(sdr, cloned);

        // Verify deep copy
        sdr.set_sparse(&[4, 5, 6]).unwrap();
        assert_ne!(sdr, cloned);
    }
}
