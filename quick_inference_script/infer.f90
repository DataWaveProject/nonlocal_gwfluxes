program infer
  use, intrinsic :: iso_fortran_env, only : sp => real32
  use ftorch

  real(kind=sp), allocatable :: input_data(:, :)
  real(kind=sp), allocatable :: ref_data(:, :)
  real(kind=sp), allocatable :: prediction(:, :)

  ! ftorch data structures
  type(torch_model) :: model
  type(torch_tensor), dimension(1) :: input_tensor, output_tensor

  integer(kind=ftorch_int), parameter :: layout(2) = [1, 2]

  logical :: success
  ! integer :: i

  call read_dataset_from_nc_2d(input_data, "input.nc")
  call read_dataset_from_nc_2d(ref_data, "predict.nc")

  allocate(prediction, mold=ref_data)

  print *, 'input_data = '
  write(*, "(5e14.5)") input_data(1:5, 1:5)
  print *, 'ref_data = '
  write(*, "(5e14.5)") ref_data(1:5, 1:5)

  ! Create Torch input/output tensors from the above arrays
  call torch_tensor_from_array(input_tensor(1), input_data, layout, torch_kCUDA)
  call torch_tensor_from_array(output_tensor(1), prediction, layout, torch_kCPU)

  ! Load ML model
  call torch_model_load(model, "saved_nlgw_model_gpu.pt", device_type=torch_kCUDA, device_index=0)

  ! Infer
  ! do i = 1, 100
    call torch_model_forward(model, input_tensor, output_tensor)
  ! end do

  print *, 'prediction = '
  write(*, "(5e14.5)") prediction(1:5, 1:5)

  print *, 'max diff = maxval(abs(prediction - ref_data))'
  print *, maxval(abs(prediction - ref_data))

  success = assert_allclose_real32_2d(prediction, ref_data, "check", 1e-1)

contains
  subroutine check(status)
  use netcdf

    integer, intent ( in) :: status

    if(status /= nf90_noerr) then 
      print *, trim(nf90_strerror(status))
      stop "Stopped"
    end if
  end subroutine check

  subroutine read_dataset_from_nc_2d(dataset, filename)
  use netcdf

  character (len = *), intent(in) :: filename
  real(kind=sp), allocatable, intent(out) :: dataset(:, :)

  real(kind=sp), allocatable :: temp(:, :)

  ! We are reading 2D data, a 6 x 12 grid. 
  integer :: dimid_0, dimid_1
  integer :: dim_0, dim_1

  ! This will be the netCDF ID for the file and data variable.
  integer :: ncid, varid

  ! Loop indexes, and error handling.
  integer :: x, y

  print *, 'reading file :: ', filename

  call check( nf90_open(filename, NF90_NOWRITE, ncid) )

  call check( nf90_inq_dimid(ncid, "dim_0", dimid_0) )
  call check( nf90_inq_dimid(ncid, "dim_1", dimid_1) )

  call check( nf90_inquire_dimension(ncid, dimid_0, len = dim_0) )
  call check( nf90_inquire_dimension(ncid, dimid_1, len = dim_1) )

  allocate(temp(dim_1, dim_0))
  allocate(dataset(dim_0, dim_1))

  call check( nf90_inq_varid(ncid, "__xarray_dataarray_variable__", varid) )
  call check( nf90_get_var(ncid, varid, temp) )

  dataset = transpose(temp)

  print *, "dataset shape :: ", shape(dataset)

  ! Close the file, freeing all resources.
  call check( nf90_close(ncid) )
  end subroutine read_dataset_from_nc_2d

  !> Print the result of a test to the terminal
  subroutine test_print(test_name, message, test_pass)

    character(len=*), intent(in) :: test_name  !! Name of the test being run
    character(len=*), intent(in) :: message    !! Message to print
    logical, intent(in) :: test_pass           !! Result of the assertion

    character(len=15) :: report

    if (test_pass) then
      report = char(27)//'[32m'//'PASSED'//char(27)//'[0m'
    else
      report = char(27)//'[31m'//'FAILED'//char(27)//'[0m'
    end if
    write(*, '(A, " :: [", A, "] ", A)') report, trim(test_name), trim(message)
  end subroutine test_print

  !> Asserts that two real32-valued 2D arrays coincide to a given relative tolerance
  function assert_allclose_real32_2d(got, expect, test_name, rtol, print_result) result(test_pass)

    character(len=*), intent(in) :: test_name           !! Name of the test being run
    real(kind=sp), intent(in), dimension(:,:) :: got    !! The array of values to be tested
    real(kind=sp), intent(in), dimension(:,:) :: expect !! The array of expected values
    real(kind=sp), intent(in), optional :: rtol         !! Optional relative tolerance (defaults to 1e-5)
    logical, intent(in), optional :: print_result       !! Optionally print test result to screen (defaults to .true.)

    logical :: test_pass  !! Did the assertion pass?

    character(len=80) :: message

    real(kind=sp) :: relative_error
    real(kind=sp) :: rtol_value
    integer :: shape_error
    logical :: print_result_value

    if (.not. present(rtol)) then
      rtol_value = 1.0e-5
    else
      rtol_value = rtol
    end if

    if (.not. present(print_result)) then
      print_result_value = .true.
    else
      print_result_value = print_result
    end if

    ! Check the shapes of the arrays match
    shape_error = maxval(abs(shape(got) - shape(expect)))
    test_pass = (shape_error == 0)

    if (test_pass) then
      test_pass = all(abs(got - expect) <= rtol_value * abs(expect))
      if (print_result_value) then
        write(message,'("relative tolerance = ", E11.4)') rtol_value
        call test_print(test_name, message, test_pass)
      end if
    else if (print_result_value) then
      call test_print(test_name, "Arrays have mismatching shapes.", test_pass)
    endif

  end function assert_allclose_real32_2d

end program infer
