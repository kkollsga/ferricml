use super::ArtifactError;

pub(crate) struct ArtifactCursor<'a> {
    remaining: &'a [u8],
}

impl<'a> ArtifactCursor<'a> {
    pub(crate) const fn new(bytes: &'a [u8]) -> Self {
        Self { remaining: bytes }
    }

    pub(crate) fn take(&mut self, count: usize) -> Result<&'a [u8], ArtifactError> {
        if self.remaining.len() < count {
            return Err(ArtifactError::Truncated);
        }
        let (value, remaining) = self.remaining.split_at(count);
        self.remaining = remaining;
        Ok(value)
    }

    pub(crate) fn u16(&mut self) -> Result<u16, ArtifactError> {
        Ok(u16::from_le_bytes(
            self.take(2)?.try_into().expect("exact length"),
        ))
    }

    pub(crate) fn u32(&mut self) -> Result<u32, ArtifactError> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("exact length"),
        ))
    }

    pub(crate) fn u64(&mut self) -> Result<u64, ArtifactError> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("exact length"),
        ))
    }

    pub(crate) fn f32(&mut self) -> Result<f32, ArtifactError> {
        Ok(f32::from_bits(self.u32()?))
    }

    pub(crate) fn f64(&mut self) -> Result<f64, ArtifactError> {
        Ok(f64::from_bits(self.u64()?))
    }

    pub(crate) const fn remaining(&self) -> &'a [u8] {
        self.remaining
    }

    /// Capacity for `requested` items of `stride` bytes each, clamped to what
    /// the unread bytes could actually supply.
    ///
    /// Element counts inside an artifact are attacker-controlled, so reserving
    /// a declared count directly lets a tiny artifact demand a large
    /// allocation before the first element is read. The number of bytes still
    /// unread is the one quantity a hostile writer cannot inflate, so it is
    /// what bounds the reservation. Growth still happens for a genuine
    /// artifact, whose bytes really are present.
    pub(crate) const fn bounded_capacity(&self, requested: usize, stride: usize) -> usize {
        debug_assert!(stride > 0);
        let affordable = self.remaining.len() / stride;
        if requested < affordable {
            requested
        } else {
            affordable
        }
    }

    pub(crate) const fn is_empty(&self) -> bool {
        self.remaining.is_empty()
    }
}
