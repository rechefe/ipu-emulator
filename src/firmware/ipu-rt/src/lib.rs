//! Hand-written HAL over the generated `ipu_pac` crate.

#![no_std]

pub use ipu_pac::*;

use core::ptr;

pub fn configure(dtype: u32, dstructure: u32, elu_alpha: u32) {
    unsafe {
        write_mmio(MMIO_BASE + dtype::OFFSET, dtype);
        write_mmio(MMIO_BASE + dstructure::OFFSET, dstructure);
        write_mmio(MMIO_BASE + elu_alpha::OFFSET, elu_alpha);
    }
}

pub fn load_program(src: *const u8, len: usize) {
    unsafe {
        ptr::copy_nonoverlapping(src, IMEM_BASE as *mut u8, len);
        write_mmio(MMIO_BASE + prog_len::OFFSET, (len / INST_ALIGNED_BYTES as usize) as u32);
    }
}

pub fn start() {
    unsafe {
        pulse_ctrl(ctrl::START_PULSE);
    }
}

pub fn wait_until_halted() {
    unsafe {
        wait_halted();
    }
}

pub fn read_pc() -> u32 {
    unsafe { read_mmio(MMIO_BASE + pc::OFFSET) }
}

pub fn write_pc(value: u32) {
    unsafe {
        write_mmio(MMIO_BASE + pc::OFFSET, value);
    }
}

pub fn reset() {
    unsafe {
        pulse_ctrl(ctrl::RESET_CMD_PULSE);
    }
}
