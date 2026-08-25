# Learnings — onnx-fp16-layernorm-fix

## Task 1+2: TRT 8.6 OBEY crash workaround
- OBEY_PRECISION_CONSTRAINTS on 607 layers → shapeContext.cpp assertion crash
- Fix: narrow to ["/ReduceMean","/Pow","/Sqrt","/Div"] only, drop set_output_type(), 296 layers, build OK

## Task 7 retry: Polygraphy dtype verification IMPOSSIBLE with dynamic shapes
- TRT 8.6 engine inspector JSON shows "N/A due to dynamic shapes" for ALL tensor dtypes
- Even with bound execution context (set_input_shape), the serialized engine cannot report dtype
- This is a fundamental TRT 8.6 limitation, NOT a code defect
- Alternative verification: build log shows "Forced 296 layers to FP32 precision" + Reformat layers exist
- Final verification MUST be mIoU comparison (Task 6) — runtime evidence is the only authoritative check
