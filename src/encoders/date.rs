//! Date Encoder implementation.
//!
//! The DateEncoder encodes up to 6 attributes of a timestamp value into an SDR:
//! - season: portion of the year (day of year)
//! - day_of_week: day of week (Monday=0 through Sunday=6)
//! - weekend: boolean for weekend detection
//! - holiday: boolean for holiday detection with smooth transitions
//! - time_of_day: time within the day
//! - custom_days: custom day-of-week categories

use crate::encoders::scalar::{ScalarEncoder, ScalarEncoderParams};
use crate::encoders::Encoder;
use crate::error::{MokoshError, Result};
use crate::types::{Real, Sdr, UInt};
use std::collections::HashSet;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Day of week constants (Monday = 0).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DayOfWeek {
    /// Monday (0)
    Monday = 0,
    /// Tuesday (1)
    Tuesday = 1,
    /// Wednesday (2)
    Wednesday = 2,
    /// Thursday (3)
    Thursday = 3,
    /// Friday (4)
    Friday = 4,
    /// Saturday (5)
    Saturday = 5,
    /// Sunday (6)
    Sunday = 6,
}

impl DayOfWeek {
    /// Parse a day name from a string.
    pub fn from_str(s: &str) -> Option<Self> {
        let lower = s.trim().to_lowercase();
        if lower.len() < 3 {
            return None;
        }
        match &lower[..3] {
            "mon" => Some(DayOfWeek::Monday),
            "tue" => Some(DayOfWeek::Tuesday),
            "wed" => Some(DayOfWeek::Wednesday),
            "thu" => Some(DayOfWeek::Thursday),
            "fri" => Some(DayOfWeek::Friday),
            "sat" => Some(DayOfWeek::Saturday),
            "sun" => Some(DayOfWeek::Sunday),
            _ => None,
        }
    }

    /// Convert from tm_wday (Sunday=0) to our format (Monday=0).
    fn from_tm_wday(wday: u32) -> Self {
        match (wday + 6) % 7 {
            0 => DayOfWeek::Monday,
            1 => DayOfWeek::Tuesday,
            2 => DayOfWeek::Wednesday,
            3 => DayOfWeek::Thursday,
            4 => DayOfWeek::Friday,
            5 => DayOfWeek::Saturday,
            _ => DayOfWeek::Sunday,
        }
    }
}

/// Holiday specification.
///
/// Holidays can be either:
/// - Annual: repeats every year on the same month/day (e.g., Christmas: month=12, day=25)
/// - One-time: occurs on a specific year/month/day (e.g., year=2018, month=4, day=1)
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Holiday {
    /// Year (None for annual holidays).
    pub year: Option<i32>,
    /// Month (1-12).
    pub month: u32,
    /// Day of month (1-31).
    pub day: u32,
}

impl Holiday {
    /// Creates an annual holiday (same date every year).
    pub fn annual(month: u32, day: u32) -> Self {
        Self {
            year: None,
            month,
            day,
        }
    }

    /// Creates a one-time holiday on a specific date.
    pub fn once(year: i32, month: u32, day: u32) -> Self {
        Self {
            year: Some(year),
            month,
            day,
        }
    }
}

/// Parameters for creating a Date Encoder.
///
/// Each attribute can be enabled by setting its width parameter > 0.
/// The total output size is the sum of all enabled attribute sizes.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DateEncoderParams {
    /// Width (bits) for season encoding. 0 to disable.
    pub season_width: UInt,
    /// Radius for season buckets (days per season). Default: 91.5 (4 seasons/year).
    pub season_radius: Real,

    /// Width (bits) for day of week encoding. 0 to disable.
    pub day_of_week_width: UInt,
    /// Radius for day of week buckets. Default: 1.0 (every day is a bucket).
    pub day_of_week_radius: Real,

    /// Width (bits) for weekend encoding. 0 to disable.
    pub weekend_width: UInt,

    /// Width (bits) for holiday encoding. 0 to disable.
    pub holiday_width: UInt,
    /// List of holidays. Default: Christmas (Dec 25).
    pub holiday_dates: Vec<Holiday>,

    /// Width (bits) for time of day encoding. 0 to disable.
    pub time_of_day_width: UInt,
    /// Radius for time of day buckets (hours). Default: 4.0 (6 periods/day).
    pub time_of_day_radius: Real,

    /// Width (bits) for custom days encoding. 0 to disable.
    pub custom_width: UInt,
    /// Custom days specification. Each string can be a day name or comma-separated list.
    /// E.g., `["Monday", "Mon,Wed,Fri"]`.
    pub custom_days: Vec<String>,
}

impl Default for DateEncoderParams {
    fn default() -> Self {
        Self {
            season_width: 0,
            season_radius: 91.5,
            day_of_week_width: 0,
            day_of_week_radius: 1.0,
            weekend_width: 0,
            holiday_width: 0,
            holiday_dates: vec![Holiday::annual(12, 25)], // Christmas
            time_of_day_width: 0,
            time_of_day_radius: 4.0,
            custom_width: 0,
            custom_days: Vec::new(),
        }
    }
}

/// A timestamp broken down into components.
#[derive(Debug, Clone, Copy)]
pub struct DateTime {
    /// Year.
    pub year: i32,
    /// Month (1-12).
    pub month: u32,
    /// Day of month (1-31).
    pub day: u32,
    /// Hour (0-23).
    pub hour: u32,
    /// Minute (0-59).
    pub minute: u32,
    /// Second (0-59).
    pub second: u32,
    /// Day of year (0-365).
    pub day_of_year: u32,
    /// Day of week (0=Sunday, 6=Saturday for tm_wday).
    pub day_of_week: u32,
}

impl DateTime {
    /// Creates a DateTime from components.
    pub fn new(
        year: i32,
        month: u32,
        day: u32,
        hour: u32,
        minute: u32,
        second: u32,
    ) -> Self {
        let day_of_year = Self::compute_day_of_year(year, month, day);
        let day_of_week = Self::compute_day_of_week(year, month, day);
        Self {
            year,
            month,
            day,
            hour,
            minute,
            second,
            day_of_year,
            day_of_week,
        }
    }

    /// Creates a DateTime from a Unix timestamp (seconds since epoch).
    pub fn from_timestamp(timestamp: i64) -> Self {
        // Simple timestamp conversion (no timezone handling)
        let days_since_epoch = timestamp / 86400;
        let time_in_day = (timestamp % 86400) as u32;

        let hour = time_in_day / 3600;
        let minute = (time_in_day % 3600) / 60;
        let second = time_in_day % 60;

        // Calculate date from days since epoch (1970-01-01)
        // This is a simplified calculation
        let (year, month, day, day_of_year) = Self::days_to_ymd(days_since_epoch);

        // Day of week: 1970-01-01 was Thursday (4)
        let day_of_week = ((days_since_epoch % 7 + 4) % 7) as u32;

        Self {
            year,
            month,
            day,
            hour,
            minute,
            second,
            day_of_year,
            day_of_week,
        }
    }

    fn is_leap_year(year: i32) -> bool {
        (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
    }

    fn days_in_month(year: i32, month: u32) -> u32 {
        match month {
            1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
            4 | 6 | 9 | 11 => 30,
            2 => {
                if Self::is_leap_year(year) {
                    29
                } else {
                    28
                }
            }
            _ => 30,
        }
    }

    fn compute_day_of_year(year: i32, month: u32, day: u32) -> u32 {
        let mut day_of_year = day - 1;
        for m in 1..month {
            day_of_year += Self::days_in_month(year, m);
        }
        day_of_year
    }

    fn compute_day_of_week(year: i32, month: u32, day: u32) -> u32 {
        // Zeller's congruence
        let mut y = year;
        let mut m = month as i32;
        if m < 3 {
            m += 12;
            y -= 1;
        }
        let q = day as i32;
        let k = y % 100;
        let j = y / 100;
        let h = (q + (13 * (m + 1)) / 5 + k + k / 4 + j / 4 - 2 * j) % 7;
        ((h + 6) % 7) as u32 // Convert to Sunday=0 format
    }

    fn days_to_ymd(mut days: i64) -> (i32, u32, u32, u32) {
        // Start from 1970
        let mut year = 1970i32;

        // Handle negative days (before 1970)
        while days < 0 {
            year -= 1;
            let days_in_year = if Self::is_leap_year(year) { 366 } else { 365 };
            days += days_in_year;
        }

        // Handle positive days
        loop {
            let days_in_year = if Self::is_leap_year(year) { 366 } else { 365 };
            if days < days_in_year {
                break;
            }
            days -= days_in_year;
            year += 1;
        }

        let day_of_year = days as u32;

        // Find month and day
        let mut month = 1u32;
        let mut remaining = days as u32;
        while month <= 12 {
            let days_in_month = Self::days_in_month(year, month);
            if remaining < days_in_month {
                break;
            }
            remaining -= days_in_month;
            month += 1;
        }

        (year, month, remaining + 1, day_of_year)
    }

    /// Convert to Unix timestamp.
    pub fn to_timestamp(&self) -> i64 {
        let mut days = 0i64;

        // Years from 1970
        for y in 1970..self.year {
            days += if Self::is_leap_year(y) { 366 } else { 365 };
        }
        for y in self.year..1970 {
            days -= if Self::is_leap_year(y) { 366 } else { 365 };
        }

        // Days in current year
        days += self.day_of_year as i64;

        // Time of day
        let seconds = days * 86400
            + self.hour as i64 * 3600
            + self.minute as i64 * 60
            + self.second as i64;

        seconds
    }
}

/// Encodes date/time values into SDR representations.
///
/// The DateEncoder can encode up to 6 different aspects of a timestamp:
/// - Season (time of year)
/// - Day of week
/// - Weekend flag
/// - Holiday flag (with smooth transitions)
/// - Time of day
/// - Custom day categories
///
/// Each aspect uses a ScalarEncoder internally, and the outputs are concatenated.
///
/// # Example
///
/// ```rust
/// use mokosh::encoders::{DateEncoder, DateEncoderParams, Encoder};
/// use mokosh::encoders::date::DateTime;
/// use mokosh::types::Sdr;
///
/// let encoder = DateEncoder::new(DateEncoderParams {
///     season_width: 5,
///     day_of_week_width: 2,
///     ..Default::default()
/// }).unwrap();
///
/// let dt = DateTime::new(2020, 1, 1, 12, 0, 0);
/// let sdr = encoder.encode_to_sdr(dt).unwrap();
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DateEncoder {
    /// Configuration parameters.
    params: DateEncoderParams,

    /// Season encoder (day of year).
    season_encoder: Option<ScalarEncoder>,

    /// Day of week encoder.
    day_of_week_encoder: Option<ScalarEncoder>,

    /// Weekend encoder (binary).
    weekend_encoder: Option<ScalarEncoder>,

    /// Holiday encoder (continuous 0-2).
    holiday_encoder: Option<ScalarEncoder>,

    /// Time of day encoder (0-24 hours).
    time_of_day_encoder: Option<ScalarEncoder>,

    /// Custom days encoder (binary).
    custom_days_encoder: Option<ScalarEncoder>,

    /// Set of custom days (as tm_wday values).
    custom_days_set: HashSet<u32>,

    /// Total output size.
    total_size: UInt,

    /// Output dimensions.
    dimensions: Vec<UInt>,
}

impl DateEncoder {
    /// Creates a new Date Encoder.
    pub fn new(params: DateEncoderParams) -> Result<Self> {
        let mut total_size = 0;

        // Season encoder
        let season_encoder = if params.season_width > 0 {
            let encoder = ScalarEncoder::new(ScalarEncoderParams {
                minimum: 0.0,
                maximum: 366.0,
                size: 0, // Will be computed from radius
                active_bits: params.season_width,
                radius: params.season_radius,
                clip_input: false,
                periodic: true,
                category: false,
            })?;
            total_size += encoder.size() as UInt;
            Some(encoder)
        } else {
            None
        };

        // Day of week encoder
        let day_of_week_encoder = if params.day_of_week_width > 0 {
            let encoder = ScalarEncoder::new(ScalarEncoderParams {
                minimum: 0.0,
                maximum: 7.0,
                size: 0,
                active_bits: params.day_of_week_width,
                radius: params.day_of_week_radius,
                clip_input: false,
                periodic: true,
                category: false,
            })?;
            total_size += encoder.size() as UInt;
            Some(encoder)
        } else {
            None
        };

        // Weekend encoder (binary: 0 or 1)
        let weekend_encoder = if params.weekend_width > 0 {
            let encoder = ScalarEncoder::new(ScalarEncoderParams {
                minimum: 0.0,
                maximum: 1.0,
                size: params.weekend_width * 2, // Two states
                active_bits: params.weekend_width,
                radius: 0.0,
                clip_input: false,
                periodic: false,
                category: true,
            })?;
            total_size += encoder.size() as UInt;
            Some(encoder)
        } else {
            None
        };

        // Holiday encoder (continuous 0-2 for smooth transitions)
        let holiday_encoder = if params.holiday_width > 0 {
            // Validate holiday dates
            for h in &params.holiday_dates {
                if h.month < 1 || h.month > 12 || h.day < 1 || h.day > 31 {
                    return Err(MokoshError::InvalidParameter {
                        name: "holiday_dates",
                        message: format!("Invalid holiday date: {:?}", h),
                    });
                }
            }
            let encoder = ScalarEncoder::new(ScalarEncoderParams {
                minimum: 0.0,
                maximum: 2.0,
                size: 0,
                active_bits: params.holiday_width,
                radius: 1.0, // Holiday radius is 1.0 day
                clip_input: false,
                periodic: true,
                category: false,
            })?;
            total_size += encoder.size() as UInt;
            Some(encoder)
        } else {
            None
        };

        // Time of day encoder
        let time_of_day_encoder = if params.time_of_day_width > 0 {
            let encoder = ScalarEncoder::new(ScalarEncoderParams {
                minimum: 0.0,
                maximum: 24.0,
                size: 0,
                active_bits: params.time_of_day_width,
                radius: params.time_of_day_radius,
                clip_input: false,
                periodic: true,
                category: false,
            })?;
            total_size += encoder.size() as UInt;
            Some(encoder)
        } else {
            None
        };

        // Custom days encoder
        let (custom_days_encoder, custom_days_set) = if params.custom_width > 0 {
            if params.custom_days.is_empty() {
                return Err(MokoshError::InvalidParameter {
                    name: "custom_days",
                    message: "custom_days list must not be empty when custom_width > 0".to_string(),
                });
            }

            // Parse custom days
            let mut days_set = HashSet::new();
            for day_str in &params.custom_days {
                for part in day_str.split(',') {
                    if let Some(day) = DayOfWeek::from_str(part) {
                        // Convert to tm_wday format (Sunday=0)
                        let wday = match day {
                            DayOfWeek::Sunday => 0,
                            DayOfWeek::Monday => 1,
                            DayOfWeek::Tuesday => 2,
                            DayOfWeek::Wednesday => 3,
                            DayOfWeek::Thursday => 4,
                            DayOfWeek::Friday => 5,
                            DayOfWeek::Saturday => 6,
                        };
                        days_set.insert(wday);
                    } else {
                        return Err(MokoshError::InvalidParameter {
                            name: "custom_days",
                            message: format!("Invalid day name: {}", part.trim()),
                        });
                    }
                }
            }

            let encoder = ScalarEncoder::new(ScalarEncoderParams {
                minimum: 0.0,
                maximum: 1.0,
                size: params.custom_width * 2,
                active_bits: params.custom_width,
                radius: 0.0,
                clip_input: false,
                periodic: false,
                category: true,
            })?;
            total_size += encoder.size() as UInt;
            (Some(encoder), days_set)
        } else {
            (None, HashSet::new())
        };

        if total_size == 0 {
            return Err(MokoshError::InvalidParameter {
                name: "params",
                message: "At least one encoder attribute must be enabled".to_string(),
            });
        }

        Ok(Self {
            params,
            season_encoder,
            day_of_week_encoder,
            weekend_encoder,
            holiday_encoder,
            time_of_day_encoder,
            custom_days_encoder,
            custom_days_set,
            total_size,
            dimensions: vec![total_size],
        })
    }

    /// Returns the parameters.
    pub fn params(&self) -> &DateEncoderParams {
        &self.params
    }

    /// Checks if a datetime is a weekend.
    ///
    /// Weekend is defined as: Friday evening (after 6pm), Saturday, and Sunday.
    fn is_weekend(dt: &DateTime) -> bool {
        dt.day_of_week == 0 // Sunday
            || dt.day_of_week == 6 // Saturday
            || (dt.day_of_week == 5 && dt.hour >= 18) // Friday evening
    }

    /// Computes the holiday value for a datetime.
    ///
    /// Returns:
    /// - 0.0: Not a holiday
    /// - 0.0-1.0: Day before holiday (ramping up)
    /// - 1.0: On the holiday
    /// - 1.0-2.0: Day after holiday (ramping down)
    fn compute_holiday_value(&self, dt: &DateTime) -> Real {
        const SECONDS_PER_DAY: i64 = 86400;
        let input_ts = dt.to_timestamp();

        for h in &self.params.holiday_dates {
            let holiday_year = h.year.unwrap_or(dt.year);
            let holiday_dt = DateTime::new(holiday_year, h.month, h.day, 0, 0, 0);
            let holiday_ts = holiday_dt.to_timestamp();

            if input_ts >= holiday_ts {
                // Holiday is in the past or now
                let diff = input_ts - holiday_ts;
                if diff < SECONDS_PER_DAY {
                    // On the holiday itself
                    return 1.0;
                } else if diff < SECONDS_PER_DAY * 2 {
                    // Day after holiday, ramp from 1 to 2
                    return 1.0 + ((diff - SECONDS_PER_DAY) as Real / SECONDS_PER_DAY as Real);
                }
            } else {
                // Holiday is in the future
                let diff = holiday_ts - input_ts;
                if diff < SECONDS_PER_DAY {
                    // Day before holiday, ramp from 0 to 1
                    return 1.0 - (diff as Real / SECONDS_PER_DAY as Real);
                }
            }
        }

        0.0
    }

    /// Encodes a DateTime into an SDR.
    pub fn encode_datetime(&self, dt: DateTime, output: &mut Sdr) -> Result<()> {
        if output.dimensions() != self.dimensions.as_slice() {
            return Err(MokoshError::DimensionMismatch {
                expected: self.dimensions.clone(),
                actual: output.dimensions().to_vec(),
            });
        }

        let mut all_bits: Vec<u32> = Vec::new();
        let mut offset = 0u32;

        // Season encoding (day of year)
        if let Some(ref encoder) = self.season_encoder {
            let day_of_year = dt.day_of_year as Real;
            let sdr = encoder.encode_to_sdr(day_of_year)?;
            for bit in sdr.get_sparse() {
                all_bits.push(bit + offset);
            }
            offset += encoder.size() as u32;
        }

        // Day of week encoding
        if let Some(ref encoder) = self.day_of_week_encoder {
            // Convert from tm_wday (Sunday=0) to Monday=0
            let day_of_week = DayOfWeek::from_tm_wday(dt.day_of_week);
            let sdr = encoder.encode_to_sdr(day_of_week as u32 as Real)?;
            for bit in sdr.get_sparse() {
                all_bits.push(bit + offset);
            }
            offset += encoder.size() as u32;
        }

        // Weekend encoding
        if let Some(ref encoder) = self.weekend_encoder {
            let val = if Self::is_weekend(&dt) { 1.0 } else { 0.0 };
            let sdr = encoder.encode_to_sdr(val)?;
            for bit in sdr.get_sparse() {
                all_bits.push(bit + offset);
            }
            offset += encoder.size() as u32;
        }

        // Custom days encoding
        if let Some(ref encoder) = self.custom_days_encoder {
            let val = if self.custom_days_set.contains(&dt.day_of_week) {
                1.0
            } else {
                0.0
            };
            let sdr = encoder.encode_to_sdr(val)?;
            for bit in sdr.get_sparse() {
                all_bits.push(bit + offset);
            }
            offset += encoder.size() as u32;
        }

        // Holiday encoding
        if let Some(ref encoder) = self.holiday_encoder {
            let val = self.compute_holiday_value(&dt);
            let sdr = encoder.encode_to_sdr(val)?;
            for bit in sdr.get_sparse() {
                all_bits.push(bit + offset);
            }
            offset += encoder.size() as u32;
        }

        // Time of day encoding
        if let Some(ref encoder) = self.time_of_day_encoder {
            let time_of_day =
                dt.hour as Real + dt.minute as Real / 60.0 + dt.second as Real / 3600.0;
            let sdr = encoder.encode_to_sdr(time_of_day)?;
            for bit in sdr.get_sparse() {
                all_bits.push(bit + offset);
            }
        }

        all_bits.sort_unstable();
        output.set_sparse_unchecked(all_bits);

        Ok(())
    }
}

impl Encoder<DateTime> for DateEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.total_size as usize
    }

    fn encode(&self, value: DateTime, output: &mut Sdr) -> Result<()> {
        self.encode_datetime(value, output)
    }
}

/// Convenience implementation for encoding Unix timestamps.
impl Encoder<i64> for DateEncoder {
    fn dimensions(&self) -> &[UInt] {
        &self.dimensions
    }

    fn size(&self) -> usize {
        self.total_size as usize
    }

    fn encode(&self, value: i64, output: &mut Sdr) -> Result<()> {
        let dt = DateTime::from_timestamp(value);
        self.encode_datetime(dt, output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_datetime_creation() {
        let dt = DateTime::new(2020, 1, 1, 12, 30, 45);
        assert_eq!(dt.year, 2020);
        assert_eq!(dt.month, 1);
        assert_eq!(dt.day, 1);
        assert_eq!(dt.hour, 12);
        assert_eq!(dt.minute, 30);
        assert_eq!(dt.second, 45);
        assert_eq!(dt.day_of_year, 0); // Jan 1 is day 0
    }

    #[test]
    fn test_datetime_from_timestamp() {
        // 2020-01-01 00:00:00 UTC = 1577836800
        let dt = DateTime::from_timestamp(1577836800);
        assert_eq!(dt.year, 2020);
        assert_eq!(dt.month, 1);
        assert_eq!(dt.day, 1);
        assert_eq!(dt.hour, 0);
    }

    #[test]
    fn test_day_of_week_parsing() {
        assert_eq!(DayOfWeek::from_str("Monday"), Some(DayOfWeek::Monday));
        assert_eq!(DayOfWeek::from_str("mon"), Some(DayOfWeek::Monday));
        assert_eq!(DayOfWeek::from_str("MON"), Some(DayOfWeek::Monday));
        assert_eq!(DayOfWeek::from_str("Friday"), Some(DayOfWeek::Friday));
        assert_eq!(DayOfWeek::from_str("sun"), Some(DayOfWeek::Sunday));
        assert_eq!(DayOfWeek::from_str("ab"), None);
    }

    #[test]
    fn test_create_date_encoder() {
        let encoder = DateEncoder::new(DateEncoderParams {
            season_width: 5,
            ..Default::default()
        })
        .unwrap();

        assert!(Encoder::<DateTime>::size(&encoder) > 0);
    }

    #[test]
    fn test_season_encoding() {
        let encoder = DateEncoder::new(DateEncoderParams {
            season_width: 5,
            ..Default::default()
        })
        .unwrap();

        // New Year's Day
        let dt = DateTime::new(2020, 1, 1, 0, 0, 0);
        let sdr = encoder.encode_to_sdr(dt).unwrap();
        assert_eq!(sdr.get_sum(), 5);

        // The first bits should be active (winter)
        let sparse = sdr.get_sparse();
        assert!(sparse.iter().all(|&b| b < 10));
    }

    #[test]
    fn test_day_of_week_encoding() {
        let encoder = DateEncoder::new(DateEncoderParams {
            day_of_week_width: 2,
            ..Default::default()
        })
        .unwrap();

        // Wednesday (2020-01-01 was Wednesday)
        let dt = DateTime::new(2020, 1, 1, 0, 0, 0);
        let sdr = encoder.encode_to_sdr(dt).unwrap();
        assert_eq!(sdr.get_sum(), 2);
    }

    #[test]
    fn test_weekend_encoding() {
        let encoder = DateEncoder::new(DateEncoderParams {
            weekend_width: 2,
            ..Default::default()
        })
        .unwrap();

        // Wednesday - not weekend
        let wed = DateTime::new(2020, 1, 1, 12, 0, 0);
        let sdr_wed = encoder.encode_to_sdr(wed).unwrap();

        // Sunday - weekend
        let sun = DateTime::new(2020, 1, 5, 12, 0, 0);
        let sdr_sun = encoder.encode_to_sdr(sun).unwrap();

        // Different encodings for weekend vs non-weekend
        assert_ne!(sdr_wed.get_sparse(), sdr_sun.get_sparse());
    }

    #[test]
    fn test_custom_days_encoding() {
        let encoder = DateEncoder::new(DateEncoderParams {
            custom_width: 2,
            custom_days: vec!["Monday".to_string(), "Wed,Fri".to_string()],
            ..Default::default()
        })
        .unwrap();

        // Monday - custom day
        let mon = DateTime::new(2020, 1, 6, 12, 0, 0); // 2020-01-06 was Monday
        let sdr_mon = encoder.encode_to_sdr(mon).unwrap();

        // Tuesday - not custom day
        let tue = DateTime::new(2020, 1, 7, 12, 0, 0);
        let sdr_tue = encoder.encode_to_sdr(tue).unwrap();

        assert_ne!(sdr_mon.get_sparse(), sdr_tue.get_sparse());
    }

    #[test]
    fn test_time_of_day_encoding() {
        let encoder = DateEncoder::new(DateEncoderParams {
            time_of_day_width: 4,
            time_of_day_radius: 4.0,
            ..Default::default()
        })
        .unwrap();

        // Midnight
        let midnight = DateTime::new(2020, 1, 1, 0, 0, 0);
        let sdr_midnight = encoder.encode_to_sdr(midnight).unwrap();

        // Noon
        let noon = DateTime::new(2020, 1, 1, 12, 0, 0);
        let sdr_noon = encoder.encode_to_sdr(noon).unwrap();

        // Different times should have different encodings
        assert_ne!(sdr_midnight.get_sparse(), sdr_noon.get_sparse());
    }

    #[test]
    fn test_combined_encoding() {
        let encoder = DateEncoder::new(DateEncoderParams {
            season_width: 5,
            day_of_week_width: 2,
            weekend_width: 2,
            time_of_day_width: 4,
            ..Default::default()
        })
        .unwrap();

        let dt = DateTime::new(2020, 1, 1, 12, 0, 0);
        let sdr = encoder.encode_to_sdr(dt).unwrap();

        // Should have bits from all enabled encoders
        assert_eq!(sdr.get_sum(), 5 + 2 + 2 + 4);
    }

    #[test]
    fn test_invalid_params_no_encoders() {
        let result = DateEncoder::new(DateEncoderParams::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_custom_days() {
        let result = DateEncoder::new(DateEncoderParams {
            custom_width: 2,
            custom_days: vec![], // Empty - should fail
            ..Default::default()
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_timestamp_encoding() {
        let encoder = DateEncoder::new(DateEncoderParams {
            season_width: 5,
            ..Default::default()
        })
        .unwrap();

        // 2020-01-01 00:00:00 UTC
        let sdr = encoder.encode_to_sdr(1577836800i64).unwrap();
        assert_eq!(sdr.get_sum(), 5);
    }

    #[test]
    fn test_is_weekend() {
        // Saturday
        let sat = DateTime::new(2020, 1, 4, 12, 0, 0);
        assert!(DateEncoder::is_weekend(&sat));

        // Sunday
        let sun = DateTime::new(2020, 1, 5, 12, 0, 0);
        assert!(DateEncoder::is_weekend(&sun));

        // Friday evening
        let fri_eve = DateTime::new(2020, 1, 3, 20, 0, 0);
        assert!(DateEncoder::is_weekend(&fri_eve));

        // Friday morning
        let fri_morn = DateTime::new(2020, 1, 3, 10, 0, 0);
        assert!(!DateEncoder::is_weekend(&fri_morn));

        // Wednesday
        let wed = DateTime::new(2020, 1, 1, 12, 0, 0);
        assert!(!DateEncoder::is_weekend(&wed));
    }
}
