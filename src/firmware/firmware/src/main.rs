//! Example bare-metal firmware — configure, load, start, wait.

#![no_std]
#![no_main]

use core::panic::PanicInfo;
use ipu_rt;

#[no_mangle]
pub extern "C" fn _start() -> ! {
    // Firmware is specialized at link time by embedding program/data symbols.
    // The integration harness loads program bytes via host_ctrl for parity tests.
    ipu_rt::configure(0, 128, 0);
    ipu_rt::start();
    ipu_rt::wait_until_halted();
    loop {}
}

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}
