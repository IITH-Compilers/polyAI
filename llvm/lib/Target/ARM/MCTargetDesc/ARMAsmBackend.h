//===-- ARMAsmBackend.h - ARM Assembler Backend -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_ARM_ARMASMBACKEND_H
#define LLVM_LIB_TARGET_ARM_ARMASMBACKEND_H

#include "MCTargetDesc/ARMFixupKinds.h"
#include "MCTargetDesc/ARMMCTargetDesc.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/TargetRegistry.h"

namespace llvm {

class ARMAsmBackend : public MCAsmBackend {
  // The STI from the target triple the MCAsmBackend was instantiated with
  // note that MCFragments may have a different local STI that should be
  // used in preference.
  const MCSubtargetInfo &STI;
  bool isThumbMode;    // Currently emitting Thumb code.
public:
  ARMAsmBackend(const Target &T, const MCSubtargetInfo &STI,
                support::endianness Endian)
      : MCAsmBackend(Endian), STI(STI),
        isThumbMode(STI.getTargetTriple().isThumb()) {}

  unsigned getNumFixupKinds() const override {
    return ARM::NumTargetFixupKinds;
  }

  // FIXME: this should be calculated per fragment as the STI may be
  // different.
  bool hasNOP() const { return STI.getFeatureBits()[ARM::HasV6T2Ops]; }

  Optional<MCFixupKind> getFixupKind(StringRef Name) const override;

  const MCFixupKindInfo &getFixupKindInfo(MCFixupKind Kind) const override;

  bool shouldForceRelocation(const MCAssembler &Asm, const MCFixup &Fixup,
                             const MCValue &Target) override;

  unsigned adjustFixupValue(const MCAssembler &Asm, const MCFixup &Fixup,
                            const MCValue &Target, uint64_t Value,
                            bool IsResolved, MCContext &Ctx,
                            const MCSubtargetInfo *STI) const;

  void applyFixup(const MCAssembler &Asm, const MCFixup &Fixup,
                  const MCValue &Target, MutableArrayRef<char> Data,
                  uint64_t Value, bool IsResolved,
                  const MCSubtargetInfo *STI) const override;

  unsigned getRelaxedOpcode(unsigned Op, const MCSubtargetInfo &STI) const;

  bool mayNeedRelaxation(const MCInst &Inst,
                         const MCSubtargetInfo &STI) const override;

  const char *reasonForFixupRelaxation(const MCFixup &Fixup,
                                       uint64_t Value) const;

  bool fixupNeedsRelaxation(const MCFixup &Fixup, uint64_t Value,
                            const MCRelaxableFragment *DF,
                            const MCAsmLayout &Layout) const override;

  void relaxInstruction(MCInst &Inst,
                        const MCSubtargetInfo &STI) const override;

  bool writeNopData(raw_ostream &OS, uint64_t Count) const override;

  void handleAssemblerFlag(MCAssemblerFlag Flag) override;

  unsigned getPointerSize() const { return 4; }
  bool isThumb() const { return isThumbMode; }
  void setIsThumb(bool it) { isThumbMode = it; }
};
} // end namespace llvm

#endif
