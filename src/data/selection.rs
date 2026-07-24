use super::SelectionError;

pub(super) fn validate_selection(
    indices: &[usize],
    available: usize,
) -> Result<(), SelectionError> {
    if indices.is_empty() {
        return Err(SelectionError::Empty);
    }
    if let Some((position, &index)) = indices
        .iter()
        .enumerate()
        .find(|(_, index)| **index >= available)
    {
        return Err(SelectionError::IndexOutOfBounds {
            position,
            index,
            available,
        });
    }
    Ok(())
}
