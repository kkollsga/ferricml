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

    pub(crate) fn f32(&mut self) -> Result<f32, ArtifactError> {
        Ok(f32::from_bits(self.u32()?))
    }

    pub(crate) const fn is_empty(&self) -> bool {
        self.remaining.is_empty()
    }
}
